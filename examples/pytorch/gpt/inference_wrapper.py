import os
import argparse
import signal
import subprocess
import torch.distributed as dist
import time

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--user_args', type=str, default="",
                        help='Script arguments for inference')

    args = parser.parse_args()
    print(f"python ../examples/pytorch/gpt/gpt_batch_maker.py {args.user_args}")
    args_list = args.user_args.split(" ")

    i=0
    while(i<2):
        p = subprocess.Popen(["python", "../examples/pytorch/gpt/gpt_batch_maker.py"] + args_list)
        code = p.wait()
        time.sleep(1)
        print(f"CODE IS {code}")
        # if code==0:
        #     break
        i+=1
    # try:
    #     os.system(f"python ../examples/pytorch/gpt/gpt_batch_maker.py {args.user_args}")
    # except RuntimeError:
    #     print("Got runtime error!")

if __name__ == "__main__":
    main()