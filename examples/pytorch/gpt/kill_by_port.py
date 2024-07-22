import os
import sys

def kill_all(num_peers, num_failures):
    os.system(f"pkill -P $(lsof -t -i:{29501+num_failures})")
    os.system(f"kill $(lsof -t -i:{29501+num_failures})")
    for i in range(num_peers):
        os.system(f"kill $(lsof -t -i:{51000+i})")
        os.system(f"kill $(lsof -t -i:{50050+i})")
        os.system(f"kill $(lsof -t -i:{8888+i})")

if __name__ == "__main__":
    kill_all(int(sys.argv[1]), int(sys.argv[2]))
