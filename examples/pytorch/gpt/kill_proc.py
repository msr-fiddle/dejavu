import os
import signal
import argparse
import time

def kill_proc():

    name = "api_worker"
    try:
        pids = []
        # iterating through each instance of the process
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            fields = line.split()
            # extracting Process ID from the output
            pid = fields[0]
            if 'mpirun' not in fields:
                pids.append(int(pid))
                #os.kill(int(pid), signal.SIGKILL)

        print("Process Successfully terminated")
        pids = sorted(pids)
        print(pids)
        os.kill(pids[-1], signal.SIGKILL)
        
        num_peers = len(pids)
        print(f"num_peers is {num_peers}")
        for i in range(num_peers):
            os.system(f"kill $(lsof -t -i:{51000+i})")

        print(f"Process {pids[-1]} Successfully terminated")
    except:
        print("Error Encountered while running script")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep_interval", type=int, default=1)

    args = parser.parse_args()
    for i in range(args.iterations):
        kill_proc()
        time.sleep(args.sleep_interval)
