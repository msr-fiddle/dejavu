import sys
import numpy as np

def parse(input_file, peers):
    ranks_dir = {}
    for p in range(peers):
        ranks_dir[p] = {
            'prompt': [],
            'tokens': [],
            'total': [],
        }

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            tokens = line.split(" ")
            if '[BENCHMARK]' in tokens:
                rank = int(tokens[2])
                if 'PROMPT' in tokens:
                    ranks_dir[rank]['prompt'].append(float(tokens[-2]))
                if 'ALL-TOKEN' in tokens:
                    ranks_dir[rank]['tokens'].append(float(tokens[-2]))
                if 'TOTAL' in tokens:
                    ranks_dir[rank]['total'].append(float(tokens[-2]))


    for p in range(peers):
        prompt_avg = round(np.average(np.asarray(ranks_dir[p]['prompt'][1:])),2)
        token_avg = round(np.average(np.asarray(ranks_dir[p]['tokens'][1:])),2)
        total_avg = round(np.average(np.asarray(ranks_dir[p]['total'][1:])),2)
        print(p, f"{prompt_avg},{token_avg},{total_avg}")

if __name__ == "__main__":
    parse(sys.argv[1], int(sys.argv[2]))