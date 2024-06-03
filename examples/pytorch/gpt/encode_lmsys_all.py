import argparse
import configparser
import os
import sys
import timeit
import json
import pathlib
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
parser.add_argument('--data_dir', type=str, default="",
                        help='path to the data directory')
args = parser.parse_args()

enc = encoder.get_encoder(args.vocab_file, args.merges_file)
input_lens = []
output_lens = []

def encode_seq(seq):
    data_seq = enc.encode(seq)
    return data_seq


def encode_file(parquet_file):
    df = pd.read_parquet(parquet_file, engine='pyarrow')
    print(df)

    for it, row in df.iterrows():
        if row['language'] == 'English':
            conv = row['conversation']
            for seq in conv:
                seq_tokens = encode_seq(seq['content'])
                if seq['role'] == 'user':
                    input_lens.append(len(seq_tokens))
                elif seq['role'] == 'assistant':
                    output_lens.append(len(seq_tokens))


if __name__ == "__main__":

    pfiles = [f for f in pathlib.Path(args.data_dir).iterdir() if f.is_file()]
    for pfile in pfiles:
        print(pfile)
        encode_file(pfile)

    print(len(input_lens))
    print(np.average(input_lens), max(input_lens))
    print(np.average(output_lens), max(output_lens))

    all_tokens = []
    for x,y in zip(input_lens, output_lens):
        all_tokens.append([x,y])

    with open('lmsys_tokens.json', 'w') as f:
        json.dump(all_tokens,f)