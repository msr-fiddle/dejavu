# Simulations

Under [dejavu/scripts/simulators](https://github.com/msr-fiddle/dejavu/tree/master/scripts/simulators), you can find simulators for the baseline non-disaggregated setup, and DéjàVu disaggregated setup.

To run our non-disaggregated, FasterTrasformer-based baseline with *N* machines, prompt time per-microbatch, per-stage is *p*, token time per microbatch per stage is *t* (all in ms):
Assuming a Poisson request distribution with rps *r*:

```bash

python simulator_baseline_trace.py --rps r --num_machines N --prompt_time p --token_time t --trace_file filename

```


To run DéjàVu with disaggregation, *A* machines used for prompt processing, *B* machines used for token generation.
Prompt time per-microbatch, per-stage is *p*, token time per microbatch per stage is *t*, and the time to transfer the cache from prompt to token is *c* (all in ms)
Assuming a Poisson request distribution with rps *r*:

```bash

python simulator_dv.py --rps r --num_prompt_machines A --num_token_machines B --prompt_time p --token_time t --cache_time c --trace_file filename

```

The file *filename* is a JSON file, containing a list of [*prompt_size*, *num_generated_tokens*].
You can add the *--do_traces* option in the above commands, to generation execution traces. JSON files are generated, which you can open with [Chromium](https://www.chromium.org/developers/how-tos/trace-event-profiling-tool/)).

# Reproduce simulations from Appendix B

The following instructions reproduce the DéjàVu Planner simulations from Appendix B.
We are at [dejavu/scripts/simulators](https://github.com/msr-fiddle/dejavu/tree/master/scripts/simulators)/

## Figure 17 (OPT-66B on 4-A100-80GB machines)

```bash

python simulate_all.py --trace_file ../../datasets/lmsys_tokens.json  --config_file model_info.json --model OPT-66B --mem_per_machine 320 --machine_cost 18.52 --output_csv opt66_4a100.csv

python plot_simulations.py --model OPT-66B --suffix 4a100  --input opt66_4a100.csv

python plot_simulations.py --model OPT-66B --suffix 4a100  --input opt66_4a100.csv --plot_cost --machine_cost 18.52

```

## Figure 18 (OPT-30B on 4-V100-16GB machines)

```bash

python simulate_all.py --trace_file ../../datasets/lmsys_tokens.json  --config_file model_info.json --model OPT-30B --mem_per_machine 64 --machine_cost 8 --output_csv opt30_4v100.csv

python plot_simulations.py --model OPT-30B --suffix 4v100  --input opt30_4v100.csv

python plot_simulations.py --model OPT-30B --suffix 4v100  --input opt30_4v100.csv --plot_cost --machine_cost 8

```

## Figure 19 (BLOOM-176B on 4-A100-80GB machines)

```bash

python simulate_all.py --trace_file ../../datasets/lmsys_tokens.json  --config_file model_info.json --model BLOOM-176B --mem_per_machine 320 --machine_cost 18.52 --output_csv bloom_4a100.csv

python plot_simulations.py --model BLOOM-176B --suffix 4a100  --input bloom_4a100.csv

python plot_simulations.py --model BLOOM-176B --suffix 4a100  --input bloom_4a100.csv --plot_cost --machine_cost 18.52

```