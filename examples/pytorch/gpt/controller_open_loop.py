# Controller frontend

import argparse
import configparser
import os
import sys
import time
import json
import random
import numpy as np

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.multiprocessing import set_start_method, Value, Manager, Process, Lock

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from examples.pytorch.gpt.api.api_client import ApiClient
from examples.pytorch.gpt.api.api_server import API_SERVER_PORT
from math import ceil

torch.classes.load_library(os.path.abspath('./lib/libth_transformer.so'))


def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--controller_ip', type=str, default="",
                        help='IP address to start controller')
    parser.add_argument('--port', type=int, default=1234,
        help='Port to start gRPC server')
    parser.add_argument('--num_peers', type=int, default=1,
        help='Number of workers')
    parser.add_argument('--num_prompt_peers', type=int, default=1,
        help='Number of prompt-only workers')
    parser.add_argument('--num_token_peers', type=int, default=1,
        help='Number of token generation workers')
    parser.add_argument('--tensor_parallelism', type=int, default=1,
        help='Tensor parallelism used in workers')
    parser.add_argument('--workers_ip_file', type=str, default="",
                        help='List with IP addresses of all workers')

    parser.add_argument('--ubatch_size', type=int, default=1,
        help='Microbatch size')
    parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
    parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        help='merges file.')
    parser.add_argument('--input_sizes_file', type=str, default="",
                        help='input sizes file.')
    parser.add_argument('--start_id', type=int, default=50256,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=50256,
                        help='end token id.')

    parser.add_argument('--poisson', action='store_true',
                        help='Whether to follow poisson distribution')
    parser.add_argument('--rps', type=float, default=1.0,
                        help='Requests per second (if poisson)')
    parser.add_argument('--input_timestamps_file', type=str, default="",
                        help='input timestamps file.')
    parser.add_argument('--num_requests', type=int, default=1,
                        help='Number of requests.')
    parser.add_argument('--input_len', type=int, default=1,
                        help='input sequence length to generate.')

    parser.add_argument('--with_ft', action='store_true',
                        help='Controller support for Fault Tolerance')

    parser.set_defaults(poisson=False)


    args = parser.parse_args()

    worker_ips = []
    with open(args.workers_ip_file, 'r') as f:
        lines = f.readlines()
        worker_ips = [x.rstrip() for x in lines]

    print(f"IP ADDRESSES ARE {worker_ips}")

    enc = encoder.get_encoder(args.vocab_file, args.merges_file)
    set_deterministic(42)

    start_id = args.start_id
    end_id = args.end_id
    ubatch_size = args.ubatch_size
    controller_ip = args.controller_ip
    port = args.port
    num_peers = args.num_peers
    num_prompt_peers = args.num_prompt_peers
    num_token_peers = args.num_token_peers
    num_requests = args.num_requests
    rps = args.rps
    tensor_parallelism = args.tensor_parallelism
    with_ft = args.with_ft
    num_pp_peers = num_peers//tensor_parallelism

    def get_input(input_len):
        # Inputs
        start_ids = [[1 for _ in range(input_len)]] * ubatch_size
        start_lengths = [len(ids) for ids in start_ids]
        #start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
        return start_ids, start_lengths

    def decode(tokens_outputs,start_point,end_point):
        outputs = []
        for i, (tokens) in enumerate(tokens_outputs):
            for beam_id in range(1):
                token = tokens[beam_id][start_point:end_point]  # exclude context input from the output
                output = enc.decode(token)
                outputs.append(output)
        return outputs


    if (args.input_sizes_file == ""):
        prompt_lengths = [args.input_len]*num_requests
        output_lengths = [100]*num_requests
    else:
        with open(args.input_sizes_file, 'r') as f:
            data = json.load(f)
            data = data[:num_requests]
            output_lengths = [min(max(x[1],2),950) for x in data]
            prompt_lengths = [args.input_len]*num_requests

    print(f"AVERAGE OUTPUT LENGTH {np.average(output_lengths)} TOKENS")

    if args.poisson:
        sleep_times = np.random.exponential(scale=1/rps, size=num_requests)
    elif args.input_timestamps_file != '':
        with open(args.input_timestamps_file, 'r') as f:
            sleep_times = json.load(f)
    else:
        sleep_times = [0] * num_requests

    print(sleep_times)

    controller = torch.classes.FasterTransformer.ControllerWrapper(controller_ip, port, num_peers, num_prompt_peers, num_token_peers, tensor_parallelism)
    controller.start_server(with_ft)

    pending = list(range(num_requests))
    to_schedule = list(range(num_requests))

    output_tokens = [[] for _ in range(num_requests)]
    input_ids = []
    input_lengths = []
    cur_output_lengths = []
    start_times = []

    for i,l in enumerate(prompt_lengths):
        start_ids, start_lengths = get_input(l)
        input_ids.append(start_ids)
        input_lengths.append(start_lengths)
        cur_output_lengths.append([output_lengths[i]]*ubatch_size)

    scheduled_ubatches = 0
    done = [False]*num_requests

    # give enough time for the workers to connect
    print("Waiting for workers to connect ....")
    api_clients = [ApiClient(worker_ips[x], API_SERVER_PORT + x) for x in range(num_peers)]
    start_idx = 0

    print("Waiting for workers to be ready ....")
    controller.wait_till_ready()
    print("Start scheduling!")

    total_start = time.time()

    num_batches = int(num_requests/num_pp_peers)
    to_schedule = num_batches
    done = 0

    while True:
        on_submitting = False
        try:
            for i in range(done,num_batches):
                print(f"Batch {i}, sleep for {sleep_times[i]}")
                time.sleep(sleep_times[i])
                for j in range(num_pp_peers):
                    idx = i*num_pp_peers + j
                    cur_input_ids = input_ids[idx]
                    cur_input_lengths = input_lengths[idx]
                    # TODO: run this in parallel?
                    print(f"Schedule request {idx}")
                    for api_client in api_clients:
                        if (not api_client.RunServeUbatch(cur_input_ids, cur_input_lengths, cur_output_lengths[idx], idx)):
                            on_submitting = True
                            raise RuntimeError
                    start_times.append(time.time()) # TODO: or 'sleep_times[i]' ?

            while True:
                if (not controller.wait_till_done()):
                    raise RuntimeError
                new_finished_reqs = controller.get_new_finished_reqs()
                pending = [x for x in pending if x not in new_finished_reqs]
                print(f"Num pending requests is {len(pending)}")
                if len(pending) == 0:
                    break

            if len(pending) == 0:
                break
        except RuntimeError as e:
                # reinitialize api clients
                print("AN ERROR OCCURED!")
                if on_submitting:
                    controller.wait_till_done()
                new_finished_reqs = controller.get_new_finished_reqs()
                pending = [x for x in pending if x not in new_finished_reqs]
                to_schedule = pending
                done = num_batches-int(len(to_schedule)/num_pp_peers)

    end_times = controller.get_finish_times()
    # token_times = controller.get_token_gen_times()

    # with open('token_times.json', 'w') as f:
    #     json.dump(token_times, f)

    # with open('end_times.json', 'w') as f:
    #     json.dump(end_times, f)

    latencies = []
    for i in range(num_requests):
        print(end_times[i], start_times[i])
        latencies.append(end_times[i]-start_times[i])

    print("All requests done!")
    total_time = time.time()-total_start
    th = num_requests/total_time
    print(f"Total time is {total_time} sec, Throughput is {th}")

    normalized_latencies = []
    for i in range(num_requests):
        normalized_latencies.append(latencies[i]/output_lengths[i])
    print(latencies, normalized_latencies)

    nl_array = np.array(normalized_latencies)
    print(f"p50: {np.percentile(nl_array, 50)} sec, p95: {np.percentile(nl_array, 95)} sec, p99: {np.percentile(nl_array, 99)} sec")

    with open('latencies.json', 'w') as f:
        json.dump(latencies, f)

    with open('normalized_latencies.json', 'w') as f:
        json.dump(normalized_latencies, f)


    # terminate
    controller.shutdown_server()

if __name__ == "__main__":
    main()
