import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='',
        help='model to use')
parser.add_argument('--bw', type=int, default=1,
        help='bandwidth')
parser.add_argument('--seq_len', type=int, default=1,
        help='maximum sequence length')
parser.add_argument('--fp_size', type=int, default=2,
        help='floating point size')

# num_layers, hidden_size - check this
models = {
    'gpt2-xl': (48,1600),
    'opt-13b': (40, 5120),
    'opt-66b': (64, 9216)
}

def get_meas(model, bs, bw, seq_len, fp_size, comp_time):
    size = models[model][0] * models[model][1] * bs * fp_size * 2 * seq_len
    time = size/bw
    time_ms = time * 1000
    total_time = time_ms #max(comp_time, time_ms)
    return total_time

if __name__ == "__main__":
    args = parser.parse_args()
    batch_sizes = [1,2,4,8]
    comp_times = [
                4984.91,
                5155.69,
                5334.88,
                5467.27,
                6105.41,
                7274.71,
                10361.64
    ]

    for bs, t in zip(batch_sizes, comp_times):
        time = get_meas(args.model, bs, args.bw, args.seq_len, args.fp_size, t)
        print(f"{time}")
