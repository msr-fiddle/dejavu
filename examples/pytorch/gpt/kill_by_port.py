import os
import sys
import subprocess

def kill_all(num_peers, num_failures):
    port = 29501+num_failures
    pids = []
    try:
        command = f"lsof -t -i:{port}"
        result = subprocess.check_output(command, shell=True, text=True)
        pids=result.split('\n')
        pids = [int(x) for x in pids[:-1]]
        print(pids)
    except Exception:
        pass
    for pid in pids:
        os.system(f"pkill -P {pid}")
    for pid in pids:
        os.system(f"kill -9 {pid}")

    for i in range(num_peers):
        os.system(f"kill $(lsof -t -i:{51000+i})")
        os.system(f"kill $(lsof -t -i:{50050+i})")
        os.system(f"kill $(lsof -t -i:{8888+i})")

if __name__ == "__main__":
    kill_all(int(sys.argv[1]), int(sys.argv[2]))
