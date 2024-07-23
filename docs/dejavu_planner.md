# DéjàVu Planner

The file [dejavu/scripts/planner.py](https://github.com/msr-fiddle/dejavu/blob/master/scripts/planner.py) contains the DéjàVu planner, that decides how to allocate resources for prompt processing and token generation.
It is based on the analysis and formulas found in 4.2.1 and Appendix D.

To run the planner with *D* machines, memory per GPU *m*, number of machines per node *G*, prompt length *P*, generating *N* extra tokens, microbatch size *b*, time per prompt *Y*, time per token *t*, and floating point size *fp* do:

```bash

python planner.py --num_machines D --memory_per_gpu m --num_machines_per_node G --prompt_len P --num_extra_tokens N --fp_size fp --batch_size b --token_time t --prompt_time Y

```