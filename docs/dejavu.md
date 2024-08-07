# DéjàVu

We support two ways to compile and run DéjàVu:
| Tool | Functionality |
| ----------- | ----------- |
| MPI | Disaggregation, Microbatch swapping
| Boost | Disaggregation, Fault Tolerance, Microbatch swapping

DéjàVu consists of a controller and multiple workers, as shown in the following figure

TODO: add figure

For the following configurations:
1. We expect the worker of rank 0 has password-less access to all other workers (via SSH).
2. In the following, replace *CONTROLLER_IP* with the IP address of the controller.
3. Create a */tmp/ip_info* file containing all workers' IP addresses, one IP address at each line. Note that if one machine has 2 workers (e.g. TMP=2), you need to add the IP address of the machine twice.
4. We are under the *dejavu/build* directory.

## Compiling with MPI

Note: As we were running our benchmarks in the cloud, we observe that MPI sometimes block communication of large messages over TCP/IP. To overcome this, add the following flags:

```bash

--mca btl_tcp_max_send_size 1000000000
--mca btl_sm_max_send_size 1000000000
--mca btl_tcp_progress_thread 1

```

You might also want to enforce MPI to not use specific interfaces:

```bash

--mca btl_tcp_if_exclude <interfaces seperated with comma>

```

### Compare disaggregated with non-disaggregated setups (e.g. Figure 8)

1. Run baseline:

We assume we have *N* workers, tensor parallelism degree *Y*, microbatch size *b*, and we serve *X* requests.
All requests have prompt size *P*.

```bash

cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_STREAM_SYNC=ON -DBUILD_MICROBATCH_INJECTION=ON ..

make -j12

python ../examples/pytorch/gpt/controller_open_loop.py --num_peers N --num_prompt_peers 0 --num_token_peers N --tensor_parallelism Y --controller_ip CONTROLLER_IP --workers_ip_file /tmp/ip_info --ubatch_size b  --num_requests X --input_len P # to start the controller

mpirun -n N -hostfile <hostfile> -x DEJAVU_CONTROLLER_IP=<CONTROLLER_IP> python ../examples/pytorch/gpt/api_worker_open.py --tensor_para_size=Y --prompt_pipeline_para_size=0 --token_pipeline_para_size=N//Y --ckpt_path <path_to_model> --backend mpi --weights_data_type fp16 --inference_data_type fp16 --ubatch_size b --num_requests X --input_len P

```

* To read a file trace (e.g. the LMSys dataset used in the DéjàVu paper), add `--input_sizes_file filename`  when starting the controller. The file *filename* is a JSON file, containing a list of [*prompt_size*, *num_generated_tokens*].
* To submit requests (prompts) following a Poisson distribution (with rps *r*), add `--poisson --rps r`    when starting the controller.

2. Run with disaggregation:

We assume we have *N* workers, tensor parallelism degree *Y*, microbatch size *b*, and we serve *X* requests.
We have *M* workers doing prompt processing, and *K* workers doing token generation.

```bash

cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_STREAM_SYNC=ON -DBUILD_SEPERATE_PROMPT=ON ..

make -j12

python ../examples/pytorch/gpt/controller_open_loop.py --num_peers N --num_prompt_peers M --num_token_peers K --tensor_parallelism Y --controller_ip <CONTROLLER_IP> --workers_ip_file /tmp/ip_info --ubatch_size b --num_requests X --input_len P

mpirun -n N -x DEJAVU_CONTROLLER_IP=<CONTROLLER_IP> python ../examples/pytorch/gpt/api_worker_open.py  --tensor_para_size=Y --prompt_pipeline_para_size=M//Y --token_pipeline_para_size=K//Y --backend mpi --ckpt_path <path_to_model> --weights_data_type fp16 --inference_data_type fp16 --ubatch_size b --num_requests X --input_len P

```

### Enable microbatch swapping

You can enable swapping by just adding the '--swapping' flag in the mpirun commands above (it can be enabled both with and without disaggregation). For the examples in Figure 9 of the paper we run DejaVu without disaggregation. Without microbatch swapping we serve *N* requests, with microbatch *b*. With microbatch swapping, we serve *N/2* requests with microbatch *2b* of homogeneous requests (i.e. all requests have fixed prompt size and generate the same number of tokens).

Note that microbatch swapping is not always beneficial for performance (due to large KV cache transfer overheads over PCIe), as detailed in Appendix G of the paper.

## Compiling without MPI

To enable fault tolerance, we support a non-MPI version that uses NCCL for forming distributed groups, and BOOST for CPU-CPU communication.

```bash

cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_STREAM_SYNC=ON -DBUILD_TEST_FAILURES=ON -DBUILD_WITH_BOOST=ON ..


make -j12

```

Start the controller

```bash

python ../examples/pytorch/gpt/controller_open_loop.py --num_peers N --num_prompt_peers M --num_token_peers K --tensor_parallelism Y --controller_ip <CONTROLLER_IP> --workers_ip_file /tmp/ip_info --ubatch_size b  --num_requests X --input_len P --with_ft

```

At each worker *i*:

```bash

export DEJAVU_CONTROLLER_IP=<CONTROLLER_IP>

export MASTER_ADDR=<IP of rank 0>
export MASTER_PORT=29501
export PROMPT_MASTER_ADDR=<IP of prompt rank 0>
export TOKEN_MASTER_ADDR=<IP of token rank 0>
export NCCL_COMM_ID=<IP of rank 0>:29512

python3.8 ../examples/pytorch/gpt/api_worker_open.py  --tensor_para_size=Y --prompt_pipeline_para_size=M//Y --token_pipeline_para_size=K//Y --backend nccl --ckpt_path <path_to_model> --weights_data_type fp16 --inference_data_type fp16 --ubatch_size b --num_requests X --rank i --world_size N --input_len P

```


### Test with failures

Fault-tolerance is supported in the non-MPI version of DéjàVu.
To showcase DéjàVu's behavior in case of failures, you can:

* Kill all DéjàVu-related processes in a node.
* The remaining processes will do some cleanup (can be observed from output messages)

Currently, DéjàVu supports only **static** allocation, meaning that the alive processes will wait until the failed process restarts.
* Restart the failed process by

```bash

python3.8 ../examples/pytorch/gpt/api_worker_open.py  --tensor_para_size=Y --prompt_pipeline_para_size=M//Y --token_pipeline_para_size=K//Y --backend nccl --ckpt_path <path_to_model> --weights_data_type fp16 --inference_data_type fp16 --ubatch_size b --num_requests X --rank i --world_size N --input_len P

```
changing the variables accordingly, and add the '--restart' and '--failures *F*' option, if this is the *F* inference restart.

For our experiments in Figures 10 and 11, we used background processes (replicas) for DéjàVu. For example, to test inference with 3 failures, you can start 3 background (replica) processes as shown above setting *F=1,2,3* when invoking each of the processes.