# This is supposed to receive requests from clients and batch them accordingly
# For now, we assume there is a list of requests waiting to be served, and this instance batches them and invokes GPT over them

import argparse
import configparser
import os
import sys
import timeit

import torch
from torch.nn.utils.rnn import pad_sequence

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils import gpt_decoder
from examples.pytorch.gpt.utils.parallel_gpt_dv import ParallelGPT

from utils import word_list
from utils import bloom

import time
import numpy as np
import signal



def signal_handler(sig, frame):
    print('SIGABRT RECEIVED!')
    exit(1)

def test(id):
    print(torch.distributed.is_initialized())

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--layer_num', type=int, default=24,
                        help='number of layers')
    parser.add_argument('--input_len', type=int, default=1,
                        help='input sequence length to generate.')
    parser.add_argument('--output_len', type=int, default=32,
                        help='output sequence length to generate.')
    parser.add_argument('--head_num', type=int, default=16,
                        help='head number')
    parser.add_argument('--size_per_head', type=int, default=64,
                        help='size per head')
    parser.add_argument('--vocab_size', type=int, default=50304,
                        help='vocab size')
    parser.add_argument('--beam_width', type=int, default=1,
                        help='beam width for beam search. Using sampling when beam width is 1.')
    parser.add_argument('--top_k', type=int, default=1,
                        help='top k candidate num')
    parser.add_argument('--top_p', type=float, default=0.,
                        help='top p probability threshold')
    parser.add_argument('--temperature', type=float, default=1.,
                        help='temperature')
    parser.add_argument('--len_penalty', type=float, default=0.,
                        help='len_penalty')
    parser.add_argument('--beam_search_diversity_rate', type=float, default=0.,
                        help='beam_search_diversity_rate')
    parser.add_argument('--tensor_para_size', type=int, default=1,
                        help='tensor parallel size')
    parser.add_argument('--pipeline_para_size', type=int, default=1,
                        help='pipeline parallel size')
    parser.add_argument('--ckpt_path', type=str, default='', help='path to the checkpoint file.')
    parser.add_argument('--lib_path', type=str, default='./lib/libth_transformer.so',
                        help='path to the pyt_fastertransformer dynamic lib file.')
    parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
    parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        help='merges file.')
    parser.add_argument('--start_id', type=int, default=50256,
                        help='start token id.')
    parser.add_argument('--end_id', type=int, default=50256,
                        help='end token id.')
    parser.add_argument('--ubatch_size', type=int, default=1,
                        help='microbatch size.')
    parser.add_argument('--num_ubatches', type=int, default=10,
                        help='Number of microbatches')
    parser.add_argument('--repetition_penalty', type=float, default=1.,
                        help='repetition penalty')
    parser.add_argument('--presence_penalty', type=float, default=0.,
                        help='presence penalty. Similar to repetition, but addive rather than multiplicative.')
    parser.add_argument('--min_length', type=int, default=0,
                        help='A minimum number of tokens to generate')
    parser.add_argument('--max_seq_len', type=int, default=1024,
                        help='max sequence length for position embedding table.')
    parser.add_argument('--inference_data_type', '--data_type', type=str, choices=['fp32', 'fp16', 'bf16'], default='fp32')
    parser.add_argument('--time', action='store_true',
                        help='whether or not to measure time elapsed.')
    parser.add_argument('--sample_input_file', type=str, default=None,
                        help='path to sample input file. If not set, it runs with no context inputs.')
    parser.add_argument('--sample_output_file', type=str, default=None,
                        help='path to sample output file.')
    parser.add_argument('--enable_random_seed', action='store_true',
                        help='is use the random seed for sentences in a batch.')
    parser.add_argument('--skip_end_tokens', dest='skip_end_tokens', action='store_true',
                        help='Whether to remove or not end tokens in outputs.')
    parser.add_argument('--no_detokenize', dest='detokenize', action='store_false',
                        help='Skip detokenizing output token ids.')
    parser.add_argument('--use_jieba_tokenizer', action='store_true',
                        help='use JiebaBPETokenizer as tokenizer.')
    parser.add_argument('--int8_mode', type=int, default=0, choices=[0, 1],
                        help='The level of quantization to perform.'
                             ' 0: No quantization. All computation in data_type'
                             ' 1: Quantize weights to int8, all compute occurs in fp16/bf16. Not supported when data_type is fp32')
    parser.add_argument(
        '--weights_data_type',
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help='Data type of FT checkpoint weights',
    )
    parser.add_argument('--return_cum_log_probs', type=int, default=0, choices=[0, 1, 2],
                        help='Whether to compute the cumulative log probsbility of sentences.'
                             ' 0: do not return the cumulative log probs '
                             ' 1: return the cumulative log probs of generated sequences'
                             ' 2: return the cumulative log probs of sequences')
    parser.add_argument('--shared_contexts_ratio', type=float, default=1.0,
                        help='Triggers the shared context optimization when'
                             'compact_size <= shared_contexts_ratio * batch_size'
                             'A value of 0.0 deactivate the optimization')
    parser.add_argument('--banned_words',
        type=str,
        default="",
        help='A comma separated list of tokens that should never be generated. Everything between the commas will'
             ' be tokenized and converted to token ids that will be banned.'
             ' Note that spaces before and after commas are included in tokenization.'
             ' An example highlighting this importance is that "the" and " the" are'
             ' two separate tokens some vocabularies.'
             ' Therefore, do ban a certain phrase, we would need to specify all tokens'
             ' in the vocabulary that include the phrase.'
             ' Example use: --banned_words "the, the,a,boy". This will ban the tokens "the", " the", "a" and "boy".'
             ' We can also use a pipe "|" to ban different tokens for different sentences in a batch.'
             ' Example: --banned_words "the, the|a,boy" will ban the tokens "the" and " the" in output sentence 1 and'
             ' ban the tokens "a" and "boy" in output sentence 2. When using this mode, we must specify a set of tokens to ban'
             ' for each sentence in the batch.',
    )
    parser.add_argument('--use_gpt_decoder_ops', action='store_true',
                        help='Use separate decoder FT operators instead of end-to-end model op.')
    parser.add_argument('--streaming', action='store_true',
                            help='Whether or not to stream the KV cache. The destination/mechanism to stream the cache is decided during compiling')
    parser.add_argument('--receive', action='store_true',
                                help='Whether or not to receive cache')
    parser.add_argument('--file_restart', action='store_true',
                            help='Whether or not to restart from a file')
    parser.add_argument('--swapping', action='store_true',
                            help='Whether or not to use swapping of KV cache at microbatches')
    parser.add_argument('--start_point', type=int, default=0,
                        help='Where to start from in our set of requests')
    parser.add_argument('--rank', type=int, default=0,
                        help='Pytorch rank')
    parser.add_argument('--world_size', type=int, default=1,
                        help='Total world size')
    parser.add_argument('--times', type=int, default=5,
                        help='Times to repeat the experiment')

    args = parser.parse_args()

    print(f"[BENCHMARK] CURRENT TIME IS {round(time.time() * 1000)} ms")

    # Load the C++ model into Pytorch model.
    torch.classes.load_library(os.path.abspath(args.lib_path))

    ckpt_config = configparser.ConfigParser()
    ckpt_config_path = os.path.join(args.ckpt_path, 'config.ini')

    if os.path.isfile(ckpt_config_path):
        ckpt_config.read(ckpt_config_path)

    if 'gpt' in ckpt_config.keys():
        for args_key, config_key, func in [
            ('layer_num', 'num_layer', ckpt_config.getint),
            ('max_seq_len', 'max_pos_seq_len', ckpt_config.getint),
            ('weights_data_type', 'weight_data_type', ckpt_config.get),
        ]:
            if config_key in ckpt_config['gpt'].keys():
                prev_val = args.__dict__[args_key]
                args.__dict__[args_key] = func('gpt', config_key)
                print('Loading {} from config.ini,    previous: {},    current: {}'.format(
                    args_key, prev_val, args.__dict__[args_key]))
            else:
                print('Not loading {} from config.ini'.format(args_key))
        for key in ['head_num', 'size_per_head', 'tensor_para_size']:
            if key in args.__dict__:
                prev_val = args.__dict__[key]
                args.__dict__[key] = ckpt_config.getint('gpt', key)
                print(key, args.__dict__[key])
                print('Loading {} from config.ini,    previous: {},    current: {}'.format(
                    key, prev_val, args.__dict__[key]))
            else:
                print('Not loading {} from config.ini'.format(key))
    if 'structure' in ckpt_config.keys():
        gpt_with_moe = ckpt_config.getboolean('structure', 'gpt_with_moe')
        expert_num = ckpt_config.getint('structure', 'expert_num')
        moe_layer_index_str = ckpt_config.get('structure', 'moe_layers')
        if len(moe_layer_index_str) <= 2:
            moe_layer_index = []
        else:
            moe_layer_index = [int(n) for n in moe_layer_index_str[1:-1].replace(' ', '').split(',')]
        moe_k = 1
    else:
        gpt_with_moe = False
        expert_num = 0
        moe_layer_index = []
        moe_k = 0

    gpt_with_moe = False
    layer_num = args.layer_num
    output_len = args.output_len
    head_num = args.head_num
    size_per_head = args.size_per_head
    vocab_size = args.vocab_size
    beam_width = args.beam_width
    top_k = args.top_k
    top_p = args.top_p
    temperature = args.temperature
    len_penalty = args.len_penalty
    beam_search_diversity_rate = args.beam_search_diversity_rate
    tensor_para_size = args.tensor_para_size
    pipeline_para_size = args.pipeline_para_size
    start_id = args.start_id
    end_id = args.end_id
    ubatch_size = args.ubatch_size
    max_seq_len = args.max_seq_len
    repetition_penalty = args.repetition_penalty
    presence_penalty = args.presence_penalty
    min_length = args.min_length
    weights_data_type = args.weights_data_type
    return_cum_log_probs = args.return_cum_log_probs
    return_output_length = return_cum_log_probs > 0
    shared_contexts_ratio = args.shared_contexts_ratio
    given_rank = args.rank
    given_world_size = args.world_size

    print('\n=================== Arguments ===================')
    for k, v in vars(args).items():
        print(f'{k.ljust(30, ".")}: {v}')
    print('=================================================\n')

    if args.use_jieba_tokenizer:
        from examples.pytorch.gpt.utils.tokenizer import JiebaBPETokenizer
        enc = JiebaBPETokenizer(args.vocab_file)
    else:
        enc = encoder.get_encoder(args.vocab_file, args.merges_file)
    torch.manual_seed(0)

    signal.signal(signal.SIGABRT, signal_handler)
    comm.initialize_model_parallel(given_world_size, given_rank)
    comm.init_group(tensor_para_size, pipeline_para_size)
    rank = comm.get_rank()
    device = comm.get_device()
    world_size = comm.get_world_size()

    print(f"HELLO! RANK IS {rank}, WORLD SIZE IS {world_size}, DEVICE IS {device}")

    ubatch_size = args.ubatch_size

    def get_input(input_len):
        # Inputs
        contexts = []
        if args.sample_input_file:  # conditional case

            with open(args.sample_input_file, "r") as f:
                data = f.read()
                data_seq = enc.encode(data)
                data = enc.decode(data_seq[:input_len])
                contexts = [data]*ubatch_size
                batch_size = ubatch_size

            contexts = contexts[:ubatch_size]
            start_ids = [torch.tensor(enc.encode(c), dtype=torch.int32, device=device) for c in contexts]
        else:  # unconditional case
            contexts = ['<|endoftext|>'] * ubatch_size
            start_ids = [torch.IntTensor([1 for _ in range(input_len)])] * ubatch_size

        start_lengths = [len(ids) for ids in start_ids]

        start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
        start_lengths = torch.IntTensor(start_lengths)
        return start_ids, start_lengths

    def define_bloom_model():
        startd = time.time()

        config_path = f'{args.ckpt_path}/config.ini'

        cfg = configparser.ConfigParser()
        cfg.read(config_path)
        model_name = 'gpt'
        inference_data_type = args.inference_data_type
        if inference_data_type == None:
            inference_data_type = cfg.get(model_name, "weight_data_type")
        model_args = dict(
            head_num=cfg.getint(model_name, 'head_num'),
            size_per_head=cfg.getint(model_name, "size_per_head"),
            layer_num=cfg.getint(model_name, "num_layer"),
            tensor_para_size=cfg.getint(model_name, "tensor_para_size"),
            vocab_size=cfg.getint(model_name, "vocab_size"),
            start_id=cfg.getint(model_name, "start_id"),
            end_id=cfg.getint(model_name, "end_id"),
            weights_data_type=cfg.get(model_name, "weight_data_type"),
            layernorm_eps=cfg.getfloat(model_name, 'layernorm_eps'),
            inference_data_type=inference_data_type,
            ckpt_path=args.ckpt_path,
            max_seq_len=max_seq_len)

        # update common parameters
        model_args.update(dict(
            lib_path=args.lib_path,
            pipeline_para_size=pipeline_para_size,
            shared_contexts_ratio=args.shared_contexts_ratio,
            int8_mode=args.int8_mode,
            torch_rank=rank
        ))

        model = bloom.Bloom(**model_args)


        print(f"[BENCHMARK] Defining model took {(time.time()-startd)} sec")
        return model


    def define_model():

        startd = time.time()
        gpt = ParallelGPT(head_num, size_per_head, vocab_size, start_id, end_id,
                          layer_num, args.ckpt_path, max_seq_len, tensor_para_size, pipeline_para_size,
                          lib_path=args.lib_path, inference_data_type=args.inference_data_type,
                          int8_mode=args.int8_mode, weights_data_type=weights_data_type,
                          shared_contexts_ratio=shared_contexts_ratio,
                          gpt_with_moe=gpt_with_moe,
                          expert_num=expert_num,
                          moe_k=moe_k,
                          moe_layer_index=moe_layer_index, torch_rank=rank)
        print(f"Defining model took {(time.time()-startd)*1000} ms")

        startl = time.time()

        return gpt

    def decode(gen_outputs,start_point,end_point):
        if not args.use_gpt_decoder_ops:
            if return_cum_log_probs > 0:
                tokens_batch, _, cum_log_probs = gen_outputs
            else:
                tokens_batch, cum_log_probs = gen_outputs, None
        else:
            tokens_batch = gen_outputs['output_token_ids']
            cum_log_probs = gen_outputs['cum_log_probs'] if return_cum_log_probs > 0 else None
        if cum_log_probs is not None:
            print('[INFO] Log probs of sentences:', cum_log_probs)

        outputs = []
        tokens_batch = tokens_batch.cpu().numpy()
        #print(tokens_batch)
        for i, (tokens) in enumerate(tokens_batch):
            for beam_id in range(beam_width):
                token = tokens[beam_id][start_point:end_point]  # exclude context input from the output
                if args.skip_end_tokens:
                    token = token[token != end_id]
                output = enc.decode(token) if args.detokenize else ' '.join(str(t) for t in token.tolist())
                outputs.append(output)
                #print(f'[INFO] batch {i}, beam {beam_id}:\n[Context]\n{context}\n\n[Output]\n{output}\n')
        return outputs



    batch_size = ubatch_size*args.pipeline_para_size
    if args.enable_random_seed:
        random_seed_tensor = torch.randint(0, 10000, size=[batch_size], dtype=torch.int64)
    else:
        random_seed_tensor = torch.zeros([batch_size], dtype=torch.int64)

    bad_words_list=None
    if args.banned_words:
        batch_banned_words = args.banned_words.split("|")
        banned_words = [[banned_words_for_batch] for banned_words_for_batch in batch_banned_words]
        bad_words_list = torch.tensor(word_list.to_word_list_format(banned_words, enc)).to("cuda")

    repetition_penalty_vec = None if repetition_penalty == 1. else repetition_penalty * torch.ones(ubatch_size, dtype=torch.float32)
    presence_penalty_vec   = None if presence_penalty == 0. else presence_penalty * torch.ones(ubatch_size, dtype=torch.float32)
    infer_decode_args = dict(
        beam_width=beam_width,
        top_k=top_k * torch.ones(batch_size, dtype=torch.int32),
        top_p=top_p * torch.ones(batch_size, dtype=torch.float32),
        temperature=temperature * torch.ones(batch_size, dtype=torch.float32),
        repetition_penalty=repetition_penalty_vec,
        presence_penalty=presence_penalty_vec,
        beam_search_diversity_rate=beam_search_diversity_rate * torch.ones(batch_size, dtype=torch.float32),
        len_penalty=len_penalty * torch.ones(size=[batch_size], dtype=torch.float32),
        bad_words_list=bad_words_list,
        min_length=min_length * torch.ones(size=[batch_size], dtype=torch.int32),
        random_seed=random_seed_tensor
    )

    def gpt_generate_fn(model, start_ids, start_lengths, output_len, finished=None, ubatch_ids=None):

        output_len = torch.tensor(output_len, dtype=torch.int32)
        if ubatch_ids:
            ubatch_ids = torch.tensor(ubatch_ids, dtype=torch.int32)
        print(start_ids, start_lengths, output_len)
        tokens_batch = model(start_ids,
                        start_lengths,
                        output_len,
                        restart=args.streaming and rank >= tensor_para_size,
                        streaming=args.streaming and rank < tensor_para_size,
                        swapping=args.swapping,
                        return_output_length=return_output_length,
                        return_cum_log_probs=return_cum_log_probs,
                        finished=finished,
                        ubatch_ids=ubatch_ids,
                        **infer_decode_args)
        return tokens_batch

    def get_custom_input(l, idx):
        sentences=[
            "This is a day to celebrate the life of a great man, a great",
            "James Best, best known",
            "Hello, I would like",
            "Once upon a time,",
            "People think ghost haunting is",
            "Among the most beautiful places",
            "Good morning New York,",
            "To the best of our"
        ]
        c=sentences[idx]
        contexts = [c]*ubatch_size

        contexts = contexts[:ubatch_size]
        start_ids = [torch.tensor(enc.encode(c), dtype=torch.int32, device=device) for c in contexts]

        start_lengths = [len(ids) for ids in start_ids]

        start_ids = pad_sequence(start_ids, batch_first=True, padding_value=end_id)
        start_lengths = torch.IntTensor(start_lengths)
        return start_ids, start_lengths

    if 'bloom' in args.ckpt_path:
        model = define_bloom_model()
    else:
        model = define_model()
    model.to_gpu()
    torch.distributed.barrier()

    prompt_lengths = [args.input_len]*args.times
    output_lengths = [args.output_len]*args.times

    prompt_lengths = prompt_lengths[args.start_point*args.pipeline_para_size:]
    output_lengths = output_lengths[args.start_point*args.pipeline_para_size:]

    num_ubatches = len(prompt_lengths)

    output_tokens = [[] for _ in range(num_ubatches)]
    input_ids = []
    input_lengths = []
    for i,l in enumerate(prompt_lengths):
        start_ids, start_lengths = get_input(l)#get_custom_input(l, i)
        input_ids.append(start_ids)
        input_lengths.append(start_lengths)

    cur_input_ids =  input_ids[:args.pipeline_para_size]
    cur_input_lengths = input_lengths[:args.pipeline_para_size]
    cur_output_lengths_one = output_lengths[:args.pipeline_para_size]
    cur_output_lengths = []
    for x in cur_output_lengths_one:
        for i in range(args.ubatch_size):
            cur_output_lengths.append(x)

    try:
        cur_mapping = list(range(args.pipeline_para_size))

        print(cur_input_ids, cur_input_lengths, cur_output_lengths, cur_mapping)

        finished = torch.tensor([0]*args.pipeline_para_size, dtype=torch.uint8)
        gen_outputs = gpt_generate_fn(model, cur_input_ids, cur_input_lengths, cur_output_lengths, finished, cur_mapping)
        scheduled_ubatches = args.pipeline_para_size

        done = [False]*num_ubatches
        #torch.cuda.synchronize()

        while scheduled_ubatches < num_ubatches:
            for i in range(args.pipeline_para_size):
                if (rank >= comm.get_world_size()//2):
                    print(f"Schedule batch {scheduled_ubatches}")
                if finished[i] or rank < comm.get_world_size()//2 :
                    output_tokens[cur_mapping[i]] = gen_outputs[i*args.ubatch_size:(i+1)*args.ubatch_size]
                    done[cur_mapping[i]] = True
                    cur_input_ids[i] = input_ids[scheduled_ubatches]
                    cur_input_lengths[i] = input_lengths[scheduled_ubatches]
                    for j in range(i*args.ubatch_size, (i+1)*args.ubatch_size):
                        cur_output_lengths[j] = output_lengths[scheduled_ubatches]
                    cur_mapping[i] = scheduled_ubatches
                    scheduled_ubatches += 1

                    if scheduled_ubatches >= num_ubatches:
                        break
            print(f"Schedule! UBATCH IDs is {cur_mapping}")
            finished = torch.tensor([0]*args.pipeline_para_size, dtype=torch.uint8)
            gen_outputs = gpt_generate_fn(model, cur_input_ids, cur_input_lengths, cur_output_lengths, finished, cur_mapping)
            torch.cuda.synchronize()

        #gen_outputs = gpt_generate_fn(model, cur_input_ids, cur_input_lengths, cur_output_lengths, finished)

        while True:
            done_reqs = [x for x in done if x]
            if len(done_reqs) == num_ubatches:
                break
            for i in range(args.pipeline_para_size):
                if finished[i] or rank < comm.get_world_size()//2 :
                    output_tokens[cur_mapping[i]] = gen_outputs[i*args.ubatch_size:(i+1)*args.ubatch_size]
                    done[cur_mapping[i]] = True
    except RuntimeError:
        print(f"Receive RuntimeError")
        raise RuntimeError

    # if True: #rank==comm.get_world_size() - 1:
    #     for i in range(num_ubatches):
    #         #print(output_tokens[i])
    #         print(f"Rank {rank}, Ubatch {i}, Generated sentence is {decode(output_tokens[i],0,prompt_lengths[i]+output_lengths[i])}")
    #model.cleanup()
    print(f"Return to main script! RANK {rank}")

if __name__ == "__main__":
    # for testing purposes
    os.environ["NCCL_P2P_DISABLE"] = "1"
    main()
