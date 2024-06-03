import argparse
import json
import queue
import numpy as np
import random
import math

from simulator_baseline_trace import simulate
from simulator_dv import simulate_dv

parser = argparse.ArgumentParser()
parser.add_argument('--num_machines', type=int, default=1,
                    help='number of machines')
parser.add_argument('--trace_file', type=str, default='',
                    help='input trace file')
parser.add_argument('--config_file', type=str, default='',
                    help='model config file')
parser.add_argument('--rps', type=float, default=0.0,
                    help='rps for Poisson')
parser.add_argument('--model', type=str, default='',
                    help='model to test with')

PROMPT_SIZE = 1000
MAX_SEQ_LEN = 2000
MEM_PER_MACHINE = 320

msizes = {
    'OPT-13B': 26,
    'OPT-30B': 60,
    'OPT-66B': 132,
    'OPT-175B': 350,
    'BLOOM-176B': 352
}

cache_per_batch_per_token = {
    'OPT-13B': 0.00085,
    'OPT-30B': 0.001375,
    'OPT-66B': 0.00236,
    'OPT-175B': 0.00483,
    'BLOOM-176B': 0.004
}

args = parser.parse_args()

def get_DV_machines(D, m, Y, N, t):
    Dp = (D * m * Y)/(m * Y + N * t)
    Dp = round(Dp)
    return Dp, D-Dp

def get_mem_needed_one_stage(model, pp, bs, seq_len):
    mem_needed = math.ceil(msizes[model]/pp) + cache_per_batch_per_token[model] * bs * seq_len
    print(cache_per_batch_per_token[model] * bs * seq_len)
    return mem_needed


def get_input():
    with open(args.trace_file, 'r') as f:
        trace_list = json.load(f)
    trace_list = [min(max(x[1],2),1000) for x in trace_list[:256]]  # for batch size 64
    #print(trace_list)
    return trace_list


def get_config():
    with open(args.config_file, 'r') as f:
        config = json.load(f)
        measurements = config[args.model]["measurements_per_bs"]
        prompt_times = [y[0] for x, y in measurements.items()]
        token_times = [y[1] for x, y in measurements.items()]
        batch_sizes = config[args.model]["batch_sizes"]
        return prompt_times, token_times, batch_sizes


def check_fits(pp, gb_per_machine, model, bs, seq_len):
    mem_needed = get_mem_needed_one_stage(model, pp, bs, seq_len)
    print(mem_needed, gb_per_machine)
    return mem_needed < gb_per_machine


def model_baseline_no_dp(trace_list, prompt_times, token_times, batch_sizes):
    print("Calling model_baseline_no_dp")
    total_times = []
    best_time = 1e9
    best_config = []

    for i, bs in enumerate(batch_sizes):
        if not check_fits(args.num_machines, MEM_PER_MACHINE, args.model, bs, MAX_SEQ_LEN):
            continue
        trace_list_new = []
        ratio = batch_sizes[-1] // bs
        for x in trace_list:
            for _ in range(ratio):
                trace_list_new.append(x)
        #print(trace_list_new)
        print(f"Check for Batch size {bs}, Num requests {len(trace_list_new)}")

        prompt_time = prompt_times[i] / args.num_machines
        token_time = token_times[i] / args.num_machines
        total_time = simulate(trace_list_new, args.rps, args.num_machines, prompt_time, token_time)
        total_times.append(total_time)

        if total_time < best_time:
            best_config = (bs, total_time)
            best_time = total_time

        print(f"Batch size {bs}, Num requests {len(trace_list_new)}, Prompt time {prompt_time}, Token time {token_time}, Total time {total_time} sec")
    return total_times, best_config


def model_baseline_with_dp(trace_list, prompt_times, token_times, batch_sizes):
    print("Calling model_baseline_with_dp")
    total_times = []
    best_time = 1e9
    best_config = []

    for dp in range(2, args.num_machines-1):
        if args.num_machines % dp != 0:
            continue
        pp = args.num_machines // dp
        for i, bs in enumerate(batch_sizes):
            if not check_fits(pp, MEM_PER_MACHINE, args.model, bs, MAX_SEQ_LEN):
                continue
            trace_list_new = []
            ratio = batch_sizes[-1] // bs
            for x in trace_list:
                for _ in range(ratio):
                    trace_list_new.append(x)

            print(f"Check for DP: {dp}, PP: {pp}, Batch size {bs}, Num requests {len(trace_list_new)}")

            prompt_time = prompt_times[i] / pp
            token_time = token_times[i] / pp

            # cut in parts and get max
            max_total_time = 0
            partition_size = len(trace_list_new) // dp
            for p in range(dp):
                trace_list_part = trace_list_new[p * partition_size:(p + 1) * partition_size]
                total_time = simulate(trace_list_part, args.rps, args.num_machines, prompt_time, token_time)
                max_total_time = max(max_total_time, total_time)

            total_times.append(max_total_time)

            if max_total_time < best_time:
                best_config = (dp, pp, bs, max_total_time)
                best_time = max_total_time

            print(f"DP: {dp}, PP: {pp}, Batch size {bs}, Num requests {len(trace_list_new)}, Prompt time {prompt_time}, Token time {token_time}, Total time {max_total_time} sec")
    return total_times, best_config


def model_dv(trace_list, prompt_times, token_times, batch_sizes):
    # TODO: add cache transfer time
    print("Calling model_dv")

    best_time = 1e9
    best_config = []

    total_times = []
    for pm in range(2, args.num_machines-1):
        tm = args.num_machines - pm
        for i, bs in enumerate(batch_sizes):
            if not check_fits(pm, MEM_PER_MACHINE, args.model, bs, PROMPT_SIZE):
                continue
            if not check_fits(tm, MEM_PER_MACHINE, args.model, bs, MAX_SEQ_LEN):
                continue
            trace_list_new = []
            ratio = batch_sizes[-1] // bs
            for x in trace_list:
                for _ in range(ratio):
                    trace_list_new.append(x)

            print(f"Check for PM: {pm}, TM: {tm}, Batch size {bs}, Num requests {len(trace_list_new)}")
            prompt_time = prompt_times[i] / pm
            token_time = token_times[i] / tm
            total_time = simulate_dv(trace_list_new, args.rps, pm, tm, prompt_time, token_time, 0)
            total_times.append(total_time)

            if total_time < best_time:
                best_config = (pm, tm, bs, total_time)
                best_time = total_time

            print(f"PM: {pm}, TM: {tm}, Batch size {bs}, Num requests {len(trace_list_new)}, Prompt time {prompt_time}, Token time {token_time}, Total time {total_time} sec")
    return total_times, best_config


def model_all():
    trace_list = get_input()
    prompt_times, token_times, batch_sizes = get_config()
    total_times_no_dp, best_config_no_dp = model_baseline_no_dp(trace_list, prompt_times, token_times, batch_sizes)
    total_times_with_dp, best_config_with_dp = model_baseline_with_dp(
          trace_list, prompt_times, token_times, batch_sizes)
    total_times_dv, best_config_dv = model_dv(trace_list, prompt_times, token_times, batch_sizes)
    print(f"Best config for Baseline, DP = 0: ", best_config_no_dp)
    print(f"Best config for Baseline, DP > 0: ", best_config_with_dp)
    print(f"Best config for DV: ", best_config_dv)

    # print(total_times_no_dp)


if __name__ == "__main__":
    model_all()
