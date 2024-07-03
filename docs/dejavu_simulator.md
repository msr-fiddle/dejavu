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