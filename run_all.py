import os

bs = [1,2,4,8]
ckpt_path = '../models/huggingface-models/c-model/opt-66b/2-gpu/'
model = 'opt-66b'

for b in bs:
    print(model, b)
    cmd = f"mpirun -np 4 --hostfile hostfile --mca btl_tcp_if_exclude docker0,enP20070s1,lo python3.8 ../examples/pytorch/gpt/gpt_batch_maker.py  --tensor_para_size=2 --pipeline_para_size=1 --ckpt_path {ckpt_path} --ubatch_size {b} --weights_data_type fp16 --inference_data_type fp16 > output-{model}-{b}"
    os.system(cmd)
