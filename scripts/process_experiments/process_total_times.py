import sys
import numpy as np

def get_data(input_file):
    total_times = []

    with open(input_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if 'BENCHMARK' in line and 'RANK 0' in line:
                tokens = line.split(" ")
                if 'TOTAL generation' in line and 'RANK 0' in line:
                    num = float(tokens[-2])
                    total_times.append(num)

    return total_times

if __name__ == "__main__":
    total_times = get_data(sys.argv[1])
    print(total_times)
    total_times_avg = np.average(total_times[2:])
    print(f"AVERAGE TIME FOR THE WHOLE BATCH (PROMPT PROCESSING AND TOKEN GENERATION): {total_times_avg} ms")