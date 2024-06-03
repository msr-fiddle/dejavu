# Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

FIND_PACKAGE(Protobuf REQUIRED)
add_library(th_parallel_gpt STATIC
    ParallelGptContextDecoderOp.cc ParallelGptDecoderOp.cc ParallelGptOp.cc
    WeightTransposeCalibrateQuantizeOp.cc)
set_property(TARGET th_parallel_gpt PROPERTY POSITION_INDEPENDENT_CODE  ON)

FIND_PACKAGE(Protobuf REQUIRED)
add_library(th_dv_controller STATIC ControllerWrapper.cc)
set_property(TARGET th_dv_controller PROPERTY POSITION_INDEPENDENT_CODE  ON)

if(BUILD_NCCL_SEND)
	add_definitions(-DNCCL_SEND=1)
endif()

if(BUILD_MPI_SEND)
	add_definitions(-DMPI_SEND=1)
endif()

if(BUILD_MICROBATCH_INJECTION)
    add_definitions(-DMICROBATCH_INJECTION=1)
endif()

if(BUILD_SEPERATE_PROMPT)
    add_definitions(-DSEPERATE_PROMPT=1)
endif()

if(BUILD_TEST_FAILURES)
    add_definitions(-DTEST_FAILURES=1)
endif()

if(BUILD_MICROBENCHMARKS)
    add_definitions(-DMICROBENCHMARKS=1)
endif()

if(BUILD_STREAM_FILE)
    add_definitions(-DSTREAM_FILE=1)
endif()

target_link_libraries(th_parallel_gpt PRIVATE "${TORCH_LIBRARIES}"
	ParallelGptContextDecoder ParallelGptDecoder ParallelGptDVBaseline ParallelGpt ParallelGptDVBenchmark ParallelGptDVFT -lprotobuf -lgrpc -lgrpc++
    calibrate_quantize_weight_kernels nccl_utils th_common th_utils)

target_link_libraries(th_dv_controller PRIVATE "${TORCH_LIBRARIES}" Controller th_common th_utils)
