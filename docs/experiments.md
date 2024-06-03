This file contains instructions for experiments presented in the paper.

1. Figure 14 (trace with failures)

* Setup:
    * 4 VMs with A100-80GB each
    * OPT-66B, fp16, 4-stage pipeline, microbatch size 1
    * Prompt-stages: 0, Token-stages: 4
    * Homogeneous requests: prompt size is 500, Num of generated tokens is 1000

* Baseline on Worker 0:

```bash

    mpirun -hostfile ../../hostfile -n 4 --mca btl_tcp_if_exclude docker0,enP20070s1,lo  python3.8 ../examples/pytorch/gpt/api_worker_open.py  --tensor_para_size=1 --prompt_pipeline_para_size=0 --token_pipeline_para_size=4 --ckpt_path ../models/huggingface-models/c-model/opt-66b/1-gpu/ --weights_data_type fp16 --inference_data_type fp16 --ubatch_size 1 --num_requests 100 --backend mpi

```

* DejaVu on each Worker 'i':

```bash
python3.8 ../examples/pytorch/gpt/api_worker_open.py  --tensor_para_size=1 --prompt_pipeline_para_size=0 --token_pipeline_para_size=4 --ckpt_path ../models/huggingface-models/c-model/opt-66b/1-gpu/  --weights_data_type fp16 --inference_data_type fp16 --ubatch_size 1 --num_requests 100 --rank {i} --world_size 4 --backend nccl
```