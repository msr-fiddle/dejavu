import argparse
from math import floor, ceil

one_gb = 1e9

# this is num_layers, layer_size (floats), attention_size (floats)
models = {
    'gpt2-1.5b': (48,30740800,1600),
    'opt-175b': (96, 1822916666, 12288)
}


parser = argparse.ArgumentParser()
parser.add_argument('--num_machines', type=int, default=1,
                    help='total number of machines')
parser.add_argument('--model', type=str, default='',
                    help='model name')
parser.add_argument('--memory_per_gpu', type=int, default=1,
                    help='memory per gpu (GB)')
parser.add_argument('--num_machines_per_node', type=int, default=1,
                    help='number of machines per node')
parser.add_argument('--prompt_len', type=int, default=1,
                    help='size of prompt')
parser.add_argument('--num_extra_tokens', type=int, default=1,
                    help='number of extra tokens to generate')
parser.add_argument('--fp_size', type=int, default=4,
                    help='FP size')
parser.add_argument('--batch_size', type=int, default=1,
                    help='size of prompt')
parser.add_argument('--token_time', type=int, default=1,
                    help='time per token')
parser.add_argument('--prompt_stream_overhead', type=float, default=1.0,
                    help='prompt stream overhead')
parser.add_argument('--prompt_time', type=int, default=1,
                    help='time per prompt')

def main(args):
    # prompt, mem limits

    layers, layer_size, attention_size = models[args.model]
    total_gpu_memory = args.num_machines_per_node * args.memory_per_gpu

    w0 = (layer_size * args.fp_size) / one_gb
    c0 = (args.prompt_len * args.batch_size * attention_size * args.fp_size * 2) / one_gb

    dp_min_mem = ceil(layers/floor(total_gpu_memory/(c0 + w0)))

    # token, mem limits
    k0c0 =  ((args.prompt_len + args.num_extra_tokens) * args.batch_size * attention_size * args.fp_size * 2) / one_gb
    tn = floor((total_gpu_memory - k0c0*layers)/w0)
    dt_min_mem = ceil(layers/tn)
    
    assert dp_min_mem >= 0
    assert dt_min_mem >= 0
    print(dp_min_mem, dt_min_mem)
    
    # to outperform baseline
    # D_t = D*N*t/(m*Y + N*t)
    A = (args.num_machines * args.num_extra_tokens * args.token_time)
    B = (args.prompt_stream_overhead * args.prompt_time + args.num_extra_tokens * args.token_time)
    print(A, B)
    dt = ceil(A/B)
    dp = args.num_machines - dt

    assert dt >= dt_min_mem
    assert dp >= dp_min_mem

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
