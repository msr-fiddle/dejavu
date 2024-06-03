# DéjàVu

## Overview
With DéjàVu, we aim to achieve fault-tolerant and resource-efficient serving of LLMs. We observe that distributed LLM serving is costly and often underutilizes hardware accelerators due to three key challenges:
1. Bubbles in pipeline-parallel deployments caused by the bimodal latency of prompt and token processing
2. GPU memory overprovisioning
3. Long recovery times in case of failures

DéjàVu addresses all these challenges using a versatile and efficient KV cache streaming library: DéjàVuLib. Using DéjàVuLib, we propose and implement:
1. Efficient prompt-token disaggregation to reduce pipeline bubbles
2. Microbatch swapping for efficient GPU memory management
3. State replication for fault-tolerance

DéjàVu is implemented on top of [NVIDIA FasterTransformer](https://github.com/NVIDIA/FasterTransformer). Like the original FasterTransformer implementation, it supports both tensor and pipeline parallelism.

## Supported Features - DéjàVuLib
DéjàVuLib is a library built to handle KV cache streaming to and from GPU
We support the following: (currently tested for the GPT, OPT and BLOOM models)
* Streaming of the KV cache to/from CPU memory and flushing local disk
* Streaming of KV cache to/from another GPU (in a different machine) via NCCL
* Streaming of KV cache to local CPU, and then flushing to another machine's CPU over the network, via MPI or BOOST

## Supported Features - DéjàVu
* Disaggregation of Prompt and Token processing
* Fault Tolerance support with cache replication
* Swapping to CPU for pipeline parallelism

## Documentation

1. Installation: Check [docs/install](docs/install.md)
2. DéjàVuLib documentation and microbenchmarks: Check [docs/dejavulib](docs/dejavulib.md)
3. DéjàVu serving system documentation and benchmarks: Check [docs/dejavu](docs/dejavu.md)
4. DéjàVu Planner documentation: Check [docs/dejavu_planner](docs/dejavu_planner.md)
5. DéjàVu simulator: Check [docs/dejavu_simulator](docs/dejavu_simulator.md)
6. For FasterTransformer original documentation: Check [docs/original_ft](docs/original_ft.md)

## Paper
If you use DéjàVu or DéjàVuLib in your research, please cite our paper:
```bash

@misc{strati2024dejavu,
      title={D\'ej\`aVu: KV-cache Streaming for Fast, Fault-tolerant Generative LLM Serving},
      author={Foteini Strati and Sara Mcallister and Amar Phanishayee and Jakub Tarnawski and Ana Klimovic},
      year={2024},
      eprint={2403.01876},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}

```
