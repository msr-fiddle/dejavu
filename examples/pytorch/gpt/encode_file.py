# encode csv files in the form of <prompt>,<request>

import argparse
import configparser
import os
import sys
import timeit
import json
import pandas as pd

import torch
from torch.nn.utils.rnn import pad_sequence

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
import examples.pytorch.gpt.utils.gpt_token_encoder as encoder
from examples.pytorch.gpt.utils import comm
from examples.pytorch.gpt.utils import gpt_decoder
from examples.pytorch.gpt.utils.parallel_gpt_dv import ParallelGPT

from utils import word_list
import time
import numpy as np
import signal

parser = argparse.ArgumentParser()
parser.add_argument('--vocab_file', type=str, default="../models/gpt2-vocab.json",
                        help='vocabulary file.')
parser.add_argument('--merges_file', type=str, default="../models/gpt2-merges.txt",
                        help='merges file.')
parser.add_argument('--csv_file', type=str, default="",
                        help='path to CSV file.')
parser.add_argument('--parquet_file', type=str, default="",
                        help='path to Parquet file.')
args = parser.parse_args()

enc = encoder.get_encoder(args.vocab_file, args.merges_file)
input_lens = []
output_lens = []

def encode_seq(seq):
    data_seq = enc.encode(seq)
    return data_seq

if __name__ == "__main__":

    if args.csv_file != "":
        df = pd.read_csv(args.csv_file, delimiter=',')
        print(df)

        for it,row in df.iterrows():
            input_seq = encode_seq(row['human_prompt'])
            output_seq = encode_seq(row['chatgpt_response'])
            input_lens.append(len(input_seq))
            output_lens.append(len(output_seq))

    elif args.parquet_file != "":
        df = pd.read_parquet(args.parquet_file, engine='pyarrow')
        print(df)

        for it,row in df.iterrows():
            input_seq = encode_seq(row['prompt'])
            output_seq = encode_seq(row['response'])
            input_lens.append(len(input_seq))
            output_lens.append(len(output_seq))

    print(np.average(input_lens), max(input_lens))
    print(np.average(output_lens), max(output_lens))

    all_tokens = []
    for x,y in zip(input_lens, output_lens):
        all_tokens.append([x,y])

    with open('chatgpt_instructions_tokens.json', 'w') as f:
        json.dump(all_tokens,f)