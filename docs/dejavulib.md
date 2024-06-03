# DéjàVuLib

## Description

DéjàVuLib is a library that enables KV cache streaming from the GPU to various destinations (and vice versa). It is built in a modular way as a set of primitives, which can be seen in the following table:

| Primitives | Functionality |
| ----------- | ----------- |
| stream_out, stream_in | Given a source (or destination) worker, the KV cache, and the inference setup (pipeline depths, batch sizes), find the proper destinations (or sources) for the different chunks of KV cache.  |
| scatter, gather | Given a non-contiguous region of KV cache, and a local or remote destination (or source), chunk the region to contiguous transfers and orchestrate movement |
| flush, fetch | Copy a contiguous chunk of KV cache. Local copies with CUDA, and remote copies with NCCL, MPI, or Boost are supported |

The DéjàVuLib can be found in [cache_utils.cc](https://github.com/msr-fiddle/dejavu/blob/main/src/fastertransformer/utils/cache_utils.cc).
The cache streaming is handled by the `CacheManager`. There are different types accoding to the underlying streaming mechanism.


## Microbenchmarks

### Get prompt-token times of various models (e.g. Fig 2, Appendix A)

We can take the time for prompt processing and per-token generation as below: the call below assumes we have a node with *N* GPUs (with tensor-model parallelism), and batch size *b*. It also assumes we have a prompt size of *P* tokens, and generates an output of *T* tokens. We are under the <dejavu/build> directory.

1. Run experiment

```bash

    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_STREAM_SYNC=ON -DBUILD_MICROBENCHMARKS=ON ..

    make -j12

    mpirun -n <N> python ../examples/pytorch/gpt/gpt_batch_maker.py  --tensor_para_size=<N> --pipeline_para_size=1 --ckpt_path <path_to_model> --ubatch_size <b> --input_len <P> --output_len <T> --weights_data_type fp16 --inference_data_type fp16 > <log_file>

```

2. Process the output (will output the average time for prompt processing and token generation)

```bash

    python ../scripts/process_prompt_token_times.py  <log_file>

```

### Run Microbenchmarks to compare with no streaming (e.g. Figure 25)

The following experiments quantify the possible overheads due to KV cache streaming under different scenarios. Note that in this case, we stream the whole KV cache, and we don't use pipeline parallelism (i.e. we don't have multiple in-flight micrrobatches). Thus, any overheads that can be hidden due to pipeline parallelism are not hidden here - this is stress-test DéjàVuLib, and see how much of the KV cache streaming can be overlapped with computation without pipeline parallelism.

1. Run with no streaming as in the previous example
2. Run streaming to local disk (just add the '--streaming' option in the above command):


```bash

    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_STREAM_SYNC=ON -DBUILD_MICROBENCHMARKS=ON -DBUILD_STREAM_FILE=ON ..

    make -j12

   mpirun -n <N> python ../examples/pytorch/gpt/gpt_batch_maker.py  --tensor_para_size=<N> --pipeline_para_size=1 --ckpt_path <path_to_model> --ubatch_size <b> --input_len <P> --output_len <T> --weights_data_type fp16 --inference_data_type fp16 --streaming > <log_file_disk>

```

3. Stream to remote GPU, using NCCL. For this, you would need a second machine (one machine is the sender, one the receiver). Make sure to provide the appropriate IP addresses in the hostfile.

```bash

    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_STREAM_SYNC=ON -DBUILD_MICROBENCHMARKS=ON -DBUILD_NCCL_SEND=ON ..

    make -j12

   mpirun -n <N> -hostfile <hostfile> python ../examples/pytorch/gpt/gpt_batch_maker.py  --tensor_para_size=<N/2> --pipeline_para_size=1 --ckpt_path <path_to_model> --ubatch_size <b> --input_len <P> --output_len <T> --weights_data_type fp16 --inference_data_type fp16 --streaming > <log_file_nccl>

```

4. Stream to remote CPU, using MPI. For this, you would need a second machine (one machine is the sender, one the receiver). Make sure to provide the appropriate IP addresses in the hostfile.

```bash

    cmake -DSM=xx -DCMAKE_BUILD_TYPE=Release -DBUILD_PYT=ON -DBUILD_MULTI_GPU=ON -DBUILD_STREAM_SYNC=ON -DBUILD_MICROBENCHMARKS=ON -DBUILD_MPI_SEND=ON ..

    make -j12

   mpirun -n <N> python ../examples/pytorch/gpt/gpt_batch_maker.py  --tensor_para_size=<N/2> --pipeline_para_size=1 --ckpt_path <path_to_model> --ubatch_size <b> --input_len <P> --output_len <T> --weights_data_type fp16 --inference_data_type fp16 --streaming > <log_file_mpi>

```

5. Process the output: it will give out the (average) total time for both prompt processing and total token generation for each of the experiments, so you can compare the overheads due to streaming in the end-to-end request latency.

```bash

    python ../scripts/process_total_times.py  <log_file>

```


### Run Microbenchmarks to check the effectiveness of DéjàVuLib optimizations

The above microbenchmarks test DéjàVuLib with all of the proposed optimizations. To understand the effectiveness of the individual optimizations we can run microbenchmarks enabling only part of the optimizations, as in Figure 7. This can be done by defining the environmental variable *DEJAVULIB_BASELINE*. Using the labels from Figure 7:

| Figure 7 baseline | DEJAVULIB_BASELINE |
| ----------- | ----------- |
| Naive Streaming | 0 |
| + DejaVu buffered copies | 1 |
| + DejaVu layer by layer streaming | 2 |
| DejaVu (all) | 3 |

If not set, all optimizations are by default enabled (i.e. like having *DEJAVULIB_BASELINE=3*).