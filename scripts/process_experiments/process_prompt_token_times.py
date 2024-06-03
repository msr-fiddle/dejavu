import sys
import numpy as np

prompts = []
all_tokens = []

def get_data(input_file):
    gen_tokens = []
    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'BENCHMARK' in line and 'RANK 0' in line:
                tokens = line.split(" ")
                if 'TOKEN' in line and 'TOTAL' not in line and 'ALL' not in line:
                    num = float(tokens[-2])
                    gen_tokens.append(num)
                elif 'PROMPT' in line:
                    prompt = float(tokens[-2])
                    prompts.append(prompt)
                if 'ALL-TOKEN' in line:
                    all_tokens.append(gen_tokens)
                    gen_tokens = []

    return prompts, all_tokens

if __name__ == "__main__":
    prompts, all_tokens = get_data(sys.argv[1])
    prompts = prompts[2:]
    all_tokens = [x[2:] for x in all_tokens]
    prompt_avg = np.average(np.asarray(prompts))
    all_tokens_avg = np.average([np.average(np.asarray(x)) for x in all_tokens])
    print(f"AVERAGE PROMPT: {prompt_avg} ms, AVERAGE TOKEN: {all_tokens_avg} ms")
