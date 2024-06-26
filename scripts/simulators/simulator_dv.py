import argparse
import json
import queue
import random
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--num_prompt_machines', type=int, default=1,
                    help='number of machines for prompt processing')
parser.add_argument('--num_token_machines', type=int, default=1,
                    help='number of machines for token processing')
parser.add_argument('--prompt_time', type=float, default=1.0,
                    help='time for microbatch prompt (ms)')
parser.add_argument('--cache_time', type=float, default=1.0,
                    help='time for microbatch prompt (ms)')
parser.add_argument('--token_time', type=float, default=1.0,
                    help='time for microbatch token (ms)')
parser.add_argument('--trace_file', type=str, default='',
                    help='input trace file')
parser.add_argument('--rps', type=float, default=0.0,
                    help='rps for Poisson')

class BatchInfo:
    def __init__(self, stage, batch, prompt, start_time, tokens) -> None:
        self.stage = stage
        self.batch = batch
        self.prompt = prompt
        self.start_time = start_time
        self.tokens = tokens


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

def create_trace(args):
    events = []
    with open(args.trace_file, 'r') as f:
        trace_list = json.load(f)

    trace_list = [min(max(x[1],2),1000) for x in trace_list[:500]]
    # trace_list_new = []
    # for x in trace_list:
    #     trace_list_new.append(x)
    #     trace_list_new.append(x)
    # trace_list = trace_list_new
    print('trace_list: ', len(trace_list))

    prompt_time_us = args.prompt_time * 1000
    token_time_us = args.token_time * 1000
    cache_time_us = args.cache_time * 1000

    ready_prompt_queue = []
    token_queue = []

    token_cur_time = [0]*args.num_token_machines
    req_start_times = generate_timestamps(len(trace_list), args.rps, args.num_token_machines + args.num_prompt_machines)
    req_end_times = [0]*len(trace_list)

    max_time = 0

    token_to_prompt_ratio = int(args.num_token_machines/args.num_prompt_machines)

    events.append(
        {
            "name": "process_name", "ph": "M", "pid": 0,
            "args": {
                "name" : "Prompt Pipeline, Stage 1"
            }
        }
    )

    events.append(
        {
            "name": "process_name", "ph": "M", "pid": 1,
            "args": {
                "name" : "Prompt Pipeline, Stage 2"
            }
        }
    )

    events.append(
        {
            "name": "process_name", "ph": "M", "pid": 2,
            "args": {
                "name" : "Token Pipeline, Stage 1"
            }
        }
    )

    events.append(
        {
            "name": "process_name", "ph": "M", "pid": 3,
            "args": {
                "name" : "Token Pipeline, Stage 2"
            }
        }
    )

    stime_stage1 = 0
    # 1. prompt processing
    for j in range(len(trace_list)):
        for i in range(args.num_prompt_machines):
            events.append({
                "pid": i,
                "ts": max(req_start_times[j],stime_stage1)+i*(prompt_time_us+cache_time_us),
                "dur": prompt_time_us,
                "ph":"X",
                "name": f"p{j}",
                "cname": trace_colors[i]
            })

            #for k in range(i*token_to_prompt_ratio, (i+1)*token_to_prompt_ratio):
            if i==args.num_prompt_machines-1:
                tm = max(req_start_times[j],stime_stage1)+args.num_prompt_machines*(prompt_time_us+cache_time_us)
                ready_prompt_queue.append(BatchInfo(0, j, True, tm, trace_list[j]))
                stime_stage1 = max(req_start_times[j],stime_stage1)+prompt_time_us
                #ready_prompt_queue.append(BatchInfo(0, j, False, (i+j)*prompt_time_us+prompt_time_us+cache_time_us, trace_list[j]))


    print(len(ready_prompt_queue))
    token_queue = ready_prompt_queue[:args.num_token_machines]
    #print(token_queue)
    for _ in range(args.num_token_machines):
        ready_prompt_queue.pop(0)

    next_prompt = args.num_token_machines
    while (len(token_queue) > 0):
        sleep = True
        for idx,req in enumerate(token_queue):
            if req.prompt and token_cur_time[0] < req.start_time:
                continue
            else:
                token_queue.pop(idx)
                sleep = False
                break

        if sleep:
            req = token_queue[0]
            token_queue.pop(0)
        #print('batch: ', req.batch)

        time_done = token_time_us
        time_done += req.start_time

        max_time = max(max_time, time_done)

        events.append({
            "pid": args.num_prompt_machines + req.stage,
            "ts": max(req.start_time, token_cur_time[req.stage]),
            "dur": token_time_us,
            "ph":"X",
            "name": f"t{req.batch},{trace_list[req.batch] - req.tokens}",
            "cname": trace_colors[req.stage]
        })

        token_cur_time[req.stage] = max(req.start_time, token_cur_time[req.stage])
        token_cur_time[req.stage] += token_time_us

        next_stage = (req.stage + 1) % args.num_token_machines
        new_req = None

        #print(f"next_stage: {next_stage}")

        if req.stage < args.num_token_machines - 1:
            #if req.tokens < trace_list[req.batch]: #or (req.stage % token_to_prompt_ratio == 0):
                new_req = BatchInfo(next_stage, req.batch, False, token_cur_time[req.stage], req.tokens)
                token_queue.append(new_req)
        else:
            if req.tokens > 1:
                new_req = BatchInfo(next_stage, req.batch, False, token_cur_time[req.stage], req.tokens-1)
                token_queue.append(new_req)
            else:
                req_end_times[req.batch] = token_cur_time[req.stage]
                if len(ready_prompt_queue) > 0:
                    new_req = ready_prompt_queue[:1]
                    #print(new_req[0].start_time/1000000)
                    for r in new_req:
                        token_queue.append(r)
                        ready_prompt_queue.pop(0)
                        next_prompt += 1

    #print(req_end_times[-1])

    # with open(f'../traces/dv_trace.json', 'w') as f:
    #     json.dump(events, f)

    dur = [(x-y)/1000000 for x,y in zip(req_end_times, req_start_times)]

    req_start_times_sec = [x/1000000 for x in req_start_times]
    req_end_times_sec = [x/1000000 for x in req_end_times]
    #print(req_start_times_sec, req_end_times_sec, dur)

    norm_lat = [x/y for x,y in zip(dur, trace_list)]
    print("LAT/TOKEN: ", np.median(norm_lat),max(norm_lat), min(norm_lat))

    max_time = max(req_end_times)
    print(f"Total time is {max_time/1e6} sec, thr is {len(trace_list)/(max_time/1e6)} ubatces/sec")


if __name__ == "__main__":
    args = parser.parse_args()
    create_trace(args)
