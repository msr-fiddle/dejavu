import argparse
import json
import queue
import numpy as np
import random

parser = argparse.ArgumentParser()
parser.add_argument('--num_machines', type=int, default=1,
                    help='number of machines')
parser.add_argument('--prompt_time', type=float, default=1.0,
                    help='time for microbatch prompt (ms)')
parser.add_argument('--token_time', type=float, default=1.0,
                    help='time for microbatch token (ms)')
parser.add_argument('--trace_file', type=str, default='',
                    help='input trace file')
parser.add_argument('--rps', type=float, default=0.0,
                    help='rps for Poisson')
parser.add_argument('--do_traces', action='store_true', help="Create traces of execution")


class BatchInfo:
    def __init__(self, stage, batch, prompt, start_time, tokens, initial_tokens) -> None:
        self.stage = stage
        self.batch = batch
        self.prompt = prompt
        self.start_time = start_time
        self.tokens = tokens
        self.initial_tokens = initial_tokens

# full list of colors is here: https://github.com/catapult-project/catapult/blob/master/tracing/tracing/base/color_scheme.html
trace_colors = [
    "thread_state_uninterruptible",
    "thread_state_iowait",
    "thread_state_running",
    "thread_state_runnable",
    "thread_state_unknown",
    "background_memory_dump",
    "light_memory_dump",
    "detailed_memory_dump",
    "vsync_highlight_color",
    "generic_work",
    "good",
    "bad",
    "terrible"
]

def set_deterministic(seed=42):
    random.seed(seed)
    np.random.seed(seed)


def generate_timestamps(size, rps, num_peers):
    set_deterministic(42)
    if rps>0.0:
        sleep_times = np.random.exponential(scale=1/rps, size=size)
    else:
        sleep_times = [0]*size
    sleep_times = [x*1000000 for x in sleep_times]
    batches = size//num_peers
    req_start_times = [0]*size
    start_time = 0
    for i in range(batches):
        for j in range(num_peers):
            req_start_times[i*num_peers + j] = start_time
        start_time += sleep_times[i]
    return req_start_times


def simulate(trace_list, rps, num_machines, prompt_time, token_time):

    prompt_time_us = prompt_time * 1000
    token_time_us = token_time * 1000

    req_start_times = generate_timestamps(len(trace_list), rps, num_machines)
    req_end_times = [0]*len(trace_list)

    batches = num_machines
    active_queue = []

    for i in range(batches):
        active_queue.append(BatchInfo(0, i, True, i*prompt_time_us, trace_list[i], trace_list[i]))
    trace_idx = batches

    cur_time = [0]*num_machines
    max_time = 0
    early_stops = 0

    if args.do_traces:
        events = []
        for i in range(args.num_machines):
            event_i = {
                "name": "process_name",
                "ph": "M",
                "pid": i,
                "args": {
                    "name" : f"Stage {i}",
                }
            }
            events.append(event_i)

    while (len(active_queue) > 0):
        sleep = True
        for idx,req in enumerate(active_queue):
            active_queue.pop(idx)
            sleep = False
            break

        if sleep:
            cur_time[0] = req_start_times[trace_idx-1]
            continue

        time_done = prompt_time_us if req.prompt else token_time_us
        time_done += req.start_time

        max_time = max(max_time, time_done)

        if args.do_traces:
            #add at trace
            events.append({
                "pid": req.stage,
                "ts": max(req.start_time, cur_time[req.stage]),
                "dur": prompt_time_us if req.prompt else token_time_us,
                "ph":"X",
                "name": f"p{req.batch}" if req.prompt else f"t{req.batch},{trace_list[req.batch] - req.tokens}",
                "cname": trace_colors[req.stage]
            })

        cur_time[req.stage] = max(req.start_time, cur_time[req.stage])
        cur_time[req.stage] += prompt_time_us if req.prompt else token_time_us

        next_stage = (req.stage + 1) % num_machines
        new_req = None

        if req.prompt:
            if req.stage < num_machines - 1:
                new_req = BatchInfo(next_stage, req.batch, True, cur_time[req.stage], req.tokens, req.initial_tokens)
            else:
                new_req = BatchInfo(next_stage, req.batch, False, cur_time[req.stage], req.tokens, req.initial_tokens)

        else:
            if req.stage < num_machines - 1:
                new_req = BatchInfo(next_stage, req.batch, False, cur_time[req.stage], req.tokens, req.initial_tokens)
            else:
                if req.tokens > 1:
                    new_req = BatchInfo(next_stage, req.batch, False, cur_time[req.stage], req.tokens-1, req.initial_tokens)
                else:
                    req_end_times[req.batch] = cur_time[req.stage]
                    for r in active_queue:
                        if (r.tokens > 2):
                            early_stops += 1
                            break

                    if trace_idx < len(trace_list):
                        new_req = BatchInfo(0, trace_idx, True, max(cur_time[0], req_start_times[trace_idx]), trace_list[trace_idx], trace_list[trace_idx])
                        trace_idx += 1


        if new_req is not None:
            active_queue.append(new_req)

    if args.do_traces:
        with open(f'baseline_trace.json', 'w') as f:
            json.dump(events, f)

    dur = [(x-y)/1000000 for x,y in zip(req_end_times, req_start_times)]
    req_start_times_sec = [x/1000000 for x in req_start_times]
    req_end_times_sec = [x/1000000 for x in req_end_times]

    # print(req_start_times_sec, req_end_times_sec, dur)
    norm_lat = [x/y for x,y in zip(dur, trace_list)]
    print(f"LAT/TOKEN: median: {np.median(norm_lat)}, max: {max(norm_lat)}, min: {min(norm_lat)}")

    max_time = max(req_end_times)
    print(f"------------------------------------------------ Early stops is {early_stops}")
    print(f"Total time is {max_time/1e6} sec, thr is {len(trace_list)/(max_time/1e6)} ubatces/sec")
    return max_time/1e6


if __name__ == "__main__":
    args = parser.parse_args()
    events = []
    with open(args.trace_file, 'r') as f:
        trace_list = json.load(f)

    trace_list = [min(max(x[1],2),1000) for x in trace_list[:512]]
    print(len(trace_list), np.average(trace_list))
    simulate(trace_list, args.rps, args.num_machines, args.prompt_time, args.token_time)
