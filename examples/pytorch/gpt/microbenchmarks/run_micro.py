import argparse
import configparser
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--num_peers', type=int, default=1,
        help='number of workers')
parser.add_argument('--tensor_para_size', type=int, default=1,
        help='tensor parallel size')
parser.add_argument('--pipeline_para_size', type=int, default=1,
        help='pipeline parallel size')
parser.add_argument('--ckpt_path', type=str, default='',
        help='model path')

def run():
    args = parser.parse_args()
    batch_sizes = [8]

    for bs in batch_sizes:
        print(f"RUN FOR BATCH SIZE {bs}")
        cmd = f"mpirun -n {args.num_peers} --allow-run-as-root  python ../examples/pytorch/gpt/gpt_batch_maker.py  --tensor_para_size={args.tensor_para_size} --pipeline_para_size={args.pipeline_para_size} --ckpt_path {args.ckpt_path} --weights_data_type fp16 --inference_data_type fp16 --ubatch_size {bs} --streaming > output_bs_{bs}"
        os.system(cmd)

if __name__ == "__main__":
    run()