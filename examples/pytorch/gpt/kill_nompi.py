import os
import signal
import argparse
import time

def kill_proc(rank, kill_rank, kill_all=False):

    if kill_all:
        os.system("bash /home/fot/dejavu/kill_all.sh")
        return

    name = "api_worker"
    try:
        pids = []
        # iterating through each instance of the process
        for line in os.popen("ps ax | grep " + name + " | grep -v grep"):
            fields = line.split()
            # extracting Process ID from the output
            pid = fields[0]
            pids.append(int(pid))

        pids = sorted(pids)
        print(pids)

        if kill_rank:
            pid_to_kill = pids[rank]
        else:
            for p in pids:
                os.kill(p, signal.SIGKILL)

        os.system(f"kill $(lsof -t -i:{51000+rank})")

        print(f"Process {pids[-1]} Successfully terminated")
    except:
        print("Error Encountered while running script")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--sleep_interval", type=int, default=1)

    args = parser.parse_args()
    for i in range(args.iterations):
        print(f"Iteration {i}")
        time.sleep(args.sleep_interval)
        kill_proc(1, False, True)
