import os
import sys

def kill_all(num_peers):
    for i in range(num_peers):
        os.system(f"kill $(lsof -t -i:{51000+i})")
        os.system(f"kill $(lsof -t -i:{50050+i})")
        os.system(f"kill $(lsof -t -i:{8888+i})")
    os.system(f"kill $(lsof -t -i:29501)")
    os.system(f"kill $(lsof -t -i:29601)")
    os.system(f"kill $(lsof -t -i:29512)")

if __name__ == "__main__":
    kill_all(int(sys.argv[1]))

