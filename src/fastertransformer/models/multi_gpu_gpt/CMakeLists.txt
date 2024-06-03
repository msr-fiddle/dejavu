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

cmake_minimum_required(VERSION 3.8)

if(BUILD_STREAM_SYNC)
	add_definitions(-DSTREAM_SYNC=1)
endif()

if(BUILD_NCCL_SEND)
	add_definitions(-DNCCL_SEND=1)
endif()

if(BUILD_MPI_SEND)
	add_definitions(-DMPI_SEND=1)
endif()

if(BUILD_STREAM_FILE)
	add_definitions(-DSTREAM_FILE=1)
endif()

if(BUILD_FLUSH_PIPELINE)
	add_definitions(-DFLUSH_PIPELINE=1)
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

if(BUILD_WITH_BOOST)
    add_definitions(-DWITH_BOOST=1)
endif()


add_library(Controller STATIC Controller.cc  ft_state.grpc.pb.cc ft_state.pb.cc)
set_property(TARGET Controller PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(Controller PUBLIC)

add_library(StateStreamClient STATIC StateStreamClient.cc  state_stream.grpc.pb.cc state_stream.pb.cc)
set_property(TARGET StateStreamClient PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(StateStreamClient PUBLIC)

find_package(Protobuf REQUIRED)
add_library(StateStreamServer STATIC StateStreamServer.cc)
set_property(TARGET StateStreamServer PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(StateStreamServer PUBLIC StateStreamClient)

add_library(ParallelGptDecoderLayerWeight STATIC ParallelGptDecoderLayerWeight.cc)
set_property(TARGET ParallelGptDecoderLayerWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDecoderLayerWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDecoderLayerWeight PUBLIC memory_utils calibrate_quantize_weight_kernels transpose_int8_kernels cuda_utils logger)

add_library(ParallelGptWeight STATIC ParallelGptWeight.cc)
set_property(TARGET ParallelGptWeight PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptWeight PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptWeight PUBLIC ParallelGptDecoderLayerWeight cuda_utils logger)

add_library(ParallelGptContextDecoder STATIC ParallelGptContextDecoder.cc)
set_property(TARGET ParallelGptContextDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptContextDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptContextDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelGptContextAttentionLayer layernorm_kernels
                                                add_residual_kernels bert_preprocess_kernels nccl_utils gpt_kernels tensor
                                                nvtx_utils cuda_utils logger StateStreamServer)

add_library(ParallelGptDecoder STATIC ParallelGptDecoder.cc)
set_property(TARGET ParallelGptDecoder PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDecoder PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDecoder PUBLIC -lcudart TensorParallelGeluFfnLayer TensorParallelReluFfnLayer
                                                TensorParallelDecoderSelfAttentionLayer layernorm_kernels
                                                add_residual_kernels nccl_utils tensor cuda_utils logger)


add_library(ControllerClient STATIC ControllerClient.cc  ft_state.grpc.pb.cc ft_state.pb.cc)
set_property(TARGET ControllerClient PROPERTY POSITION_INDEPENDENT_CODE  ON)
target_link_libraries(ControllerClient PUBLIC)

add_library(ParallelGpt STATIC ParallelGpt.cc)
set_property(TARGET ParallelGpt PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGpt PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGpt PUBLIC -lcudart ParallelGptDecoder ParallelGptContextDecoder StateStreamClient decoding_kernels gpt_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias ParallelGptWeight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils cache_utils -lprotobuf -lgrpc -lgrpc++)

add_library(ParallelGptDVBaseline STATIC ParallelGptDVBaseline.cc)
set_property(TARGET ParallelGptDVBaseline PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDVBaseline PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDVBaseline PUBLIC -lcudart ParallelGptDecoder ParallelGptContextDecoder StateStreamClient decoding_kernels gpt_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias ParallelGptWeight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils cache_utils -lprotobuf -lgrpc -lgrpc++)

add_library(ParallelGptDVBenchmark STATIC ParallelGptDVBenchmark.cc)
set_property(TARGET ParallelGptDVBenchmark PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDVBenchmark PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDVBenchmark PUBLIC -lcudart ParallelGptDecoder ParallelGptContextDecoder StateStreamClient ControllerClient decoding_kernels gpt_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias ParallelGptWeight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils cache_utils -lprotobuf -lgrpc -lgrpc++)

add_library(ParallelGptDVFT STATIC ParallelGptDVFT.cc)
set_property(TARGET ParallelGptDVFT PROPERTY POSITION_INDEPENDENT_CODE  ON)
set_property(TARGET ParallelGptDVFT PROPERTY CUDA_RESOLVE_DEVICE_SYMBOLS  ON)
target_link_libraries(ParallelGptDVFT PUBLIC -lcudart ParallelGptDecoder ParallelGptContextDecoder ControllerClient decoding_kernels gpt_kernels
                      DynamicDecodeLayer BaseBeamSearchLayer bert_preprocess_kernels gen_relative_pos_bias ParallelGptWeight
                      custom_ar_comm logprob_kernels cuda_utils logger nvtx_utils cache_utils -lprotobuf -lgrpc -lgrpc++)


add_executable(gpt_gemm gpt_gemm.cc)
target_link_libraries(gpt_gemm PUBLIC -lcudart gpt_gemm_func memory_utils cuda_utils logger)
