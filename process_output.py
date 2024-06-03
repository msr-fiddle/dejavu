import sys
import numpy as np

def get_data(inp):

    idx = 0
    prompts = []
    tokens = []
    totals = []

    with open(inp, 'r') as f:
        lines = f.readlines()

        for line in lines:
            if 'START GPT FORWARD BENCHMARK' in line:
                idx += 1

            if idx > 2:
                data = line.split(" ")
                if 'PROMPT processing' in line:
                    prompts.append(float(data[-2]))
                elif 'ALL-TOKEN generation' in line:
                    tokens.append(float(data[-2]))
                elif 'TOTAL generation' in line:
                    totals.append(float(data[-2]))

    return prompts, tokens, totals


if __name__ == "__main__":
    prompts, tokens, totals = get_data(sys.argv[1])
    print(f"PROMPTS: {prompts}")
    print(f"TOKENS: {tokens}")
    print(f"TOTALS: {totals}")

    pavg = round(np.average(prompts), 2)
    tokavg = round(np.average(tokens), 2)
    totavg = round(np.average(totals), 2)

    print(f"{pavg},{tokavg},{totavg}")
