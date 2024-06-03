import argparse
import configparser
import os
import sys
import timeit
import json

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
args = parser.parse_args()

enc = encoder.get_encoder(args.vocab_file, args.merges_file)
input_lens = []
output_lens = []

def encode_seq(seq):
    data_seq = enc.encode(seq)
    return data_seq

def encode_inputs(input_json):
    for d in data:
        inst_tokens = encode_seq(d['instruction'])
        input_tokens = encode_seq(d['input'])
        input_lens.append(len(input_tokens)+len(inst_tokens))

def encode_outputs(input_json):
    for d in data:
        output_tokens = encode_seq(d['output'])
        output_lens.append(len(output_tokens))

if __name__ == "__main__":
    with open('alpaca_data.json','r') as f:
        data = json.load(f)

    encode_inputs(data)
    encode_outputs(data)

print(np.average(input_lens), max(input_lens))
print(np.average(output_lens), max(output_lens))

all_tokens = []
for x,y in zip(input_lens, output_lens):
    all_tokens.append([x,y])

with open('alpaca_tokens.json', 'w') as f:
    json.dump(all_tokens,f)