/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 * Copyright (c) 2021, NAVER Corp.  Authored by CLOVA.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifdef MICROBENCHMARKS
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDVBenchmark.h"
#elif MICROBATCH_INJECTION
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDVBaseline.h"
#elif defined(TEST_FAILURES) || defined(SEPERATE_PROMPT)
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDVFT.h"
#else
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGpt.h"
#endif
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

using std::vector;

static bool grpc_server_started = false;

class IFGpt {
public:
    virtual ~IFGpt() {}
    virtual void forward(th::Tensor&               input_ids,
                         th::Tensor&               input_lengths,
                         th::Tensor&               output_ids,
                         th::Tensor&               sequence_lengths,
                         th::Tensor&               cum_log_probs,
                         const size_t              request_output_len,
                         const size_t              beam_width,
                         th::optional<th::Tensor>  reload,
                         th::optional<th::Tensor>  streaming,
                         th::optional<th::Tensor>  swapping,
                         th::optional<th::Tensor>  top_k_opt,
                         th::optional<th::Tensor>  top_p_opt,
                         th::optional<th::Tensor>  beam_search_diversity_rate_opt,
                         th::optional<th::Tensor>  temperature_opt,
                         th::optional<th::Tensor>  len_penalty_opt,
                         th::optional<th::Tensor>  repetition_penalty_opt,
                         th::optional<th::Tensor>  presence_penalty_opt,
                         th::optional<th::Tensor>  min_length_opt,
                         th::optional<th::Tensor>  random_seed_opt,
                         th::optional<th::Tensor>  bad_words_list_opt,
                         th::optional<th::Tensor>& finished_opt,
                         th::optional<th::Tensor>  ubatch_output_lengths,
                         th::optional<th::Tensor>  ubatch_ids,
                         th::optional<int64_t>     return_cum_log_probs_opt) = 0;
    virtual void cleanup()                                               = 0;
    virtual void reset()                                                 = 0;
};

template<typename T>
class FTGpt: public IFGpt {
public:
#ifdef MICROBENCHMARKS
    ft::ParallelGptDVBenchmark<T>* gpt_ptr = NULL;
#elif MICROBATCH_INJECTION
    ft::ParallelGptDVBaseline<T>* gpt_ptr = NULL;
#elif defined(TEST_FAILURES) || defined(SEPERATE_PROMPT)
    ft::ParallelGptDVFT<T>* gpt_ptr = NULL;
#else
    ft::ParallelGpt<T>* gpt_ptr = NULL;
#endif
    cudaStream_t*                         stream_ = NULL;
    ft::Allocator<ft::AllocatorType::TH>* allocator;

    FTGpt(const int64_t              head_num,
          const int64_t              size_per_head,
          const int64_t              inter_size,
          const int64_t              layer_num,
          const int64_t              expert_num,
          const int64_t              moe_k,
          const std::vector<int64_t> moe_layer_index,
          const int64_t              vocab_size,
          const ft::gptVariantParams gpt_variant_params,
          const int64_t              start_id,
          const int64_t              end_id,
          const int64_t              tensor_para_size,
          const int64_t              pipeline_para_size,
          const int64_t              int8_mode,
          const vector<th::Tensor>   weights,
          const vector<th::Tensor>   int8_weights,
          const vector<th::Tensor>   scale,
          const double               shared_contexts_ratio,
          const int64_t              prompt_world_size,
          const int64_t              token_world_size,
          const int64_t              torch_rank,
          const bool                 is_restart):
        head_num_(head_num),
        size_per_head_(size_per_head),
        inter_size_(inter_size),
        layer_num_(layer_num),
        expert_num_(expert_num),
        moe_k_(moe_k),
        moe_layer_index_(moe_layer_index),
        gpt_variant_params_(gpt_variant_params),
        vocab_size_(vocab_size),
        start_id_(start_id),
        end_id_(end_id),
        tensor_para_size_(tensor_para_size),
        pipeline_para_size_(pipeline_para_size),
        int8_mode_(int8_mode),
        weights_(weights),
        int8_weights_(int8_weights),
        scale_(scale),
        shared_contexts_ratio_(shared_contexts_ratio),
        prompt_world_size_(prompt_world_size),
        token_world_size_(token_world_size),
        torch_rank_(torch_rank),
        is_restart_(is_restart)
    {

        ft::check_cuda_error(cublasLtCreate(&cublasltHandle_));
        cublas_algo_map_      = new ft::cublasAlgoMap(GEMM_CONFIG);
        cublas_wrapper_mutex_ = new std::mutex();

        int device_id = 0;
        ft::check_cuda_error(cudaGetDevice(&device_id));
        ft::check_cuda_error(cudaGetDeviceProperties(&prop_, device_id));
        printf("-------------- Device %d\n", device_id);

        bool with_mpi = true;
#if defined(NCCL_SEND) || defined(MPI_SEND)
        int cache_para_size = 2 * tensor_para_size * pipeline_para_size;
#elif defined(TEST_FAILURES) || defined(SEPERATE_PROMPT)
        int cache_para_size = prompt_world_size + token_world_size;
#else
        int cache_para_size = tensor_para_size * pipeline_para_size_;
#endif

#if defined(TEST_FAILURES)
        with_mpi = false;
#endif

        printf("CACHE_PARA_SIZE IS %d\n", cache_para_size);
        ftNcclCacheInitialize(cache_stream_para_, cache_para_size, torch_rank_, with_mpi);

        ftNcclInitialize(tensor_para_,
                         pipeline_para_,
                         cache_stream_para_,
                         tensor_para_size,
                         pipeline_para_size,
                         prompt_world_size_,
                         token_world_size_,
                         cache_para_size,
                         torch_rank,
                         with_mpi);
        printf(
            "AFTER INITS, CACHE SIZE %d, CACHE RANK %d, TENSOR SIZE %d, TENSOR RANK %d, PIPELINE SIZE %d, PIPELINE RANK %d\n",
            cache_stream_para_.world_size_,
            cache_stream_para_.rank_,
            tensor_para_.world_size_,
            tensor_para_.rank_,
            pipeline_para_.world_size_,
            pipeline_para_.rank_);

        gpt_weights_.resizeLayer(layer_num_);
        for (int i = 0; i < (int)layer_num_; i++) {
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 0 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->pre_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 1 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.kernel =
                get_ptr<T>(weights_[i + 2 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.bias =
                get_ptr<T>(weights_[i + 3 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.kernel =
                get_ptr<T>(weights_[i + 4 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.bias =
                get_ptr<T>(weights_[i + 5 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.gamma =
                get_ptr<T>(weights_[i + 6 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->self_attn_layernorm_weights.beta =
                get_ptr<T>(weights_[i + 7 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.kernel =
                get_ptr<T>(weights_[i + 8 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.bias =
                get_ptr<T>(weights_[i + 9 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.kernel =
                get_ptr<T>(weights_[i + 10 * layer_num_]);
            gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.bias =
                get_ptr<T>(weights_[i + 11 * layer_num_]);

            if (int8_mode_ != 0) {
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 0 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 1 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 2 * layer_num_]);
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.int8_kernel =
                    get_ptr<int8_t>(int8_weights_[i + 3 * layer_num_]);

                if (int8_mode == 1) {
                    gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 0 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]
                        ->self_attention_weights.attention_output_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 1 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 2 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.weight_only_quant_scale =
                        get_ptr<T>(scale_[i + 3 * layer_num_]);
                }
                else {
                    gpt_weights_.decoder_layer_weights[i]->self_attention_weights.query_weight.scale =
                        get_ptr<float>(scale_[i + 0 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->self_attention_weights.attention_output_weight.scale =
                        get_ptr<float>(scale_[i + 1 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->ffn_weights.intermediate_weight.scale =
                        get_ptr<float>(scale_[i + 2 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->ffn_weights.output_weight.scale =
                        get_ptr<float>(scale_[i + 3 * layer_num_]);
                }
            }
        }

        size_t weight_offset = 0;
        if (gpt_variant_params_.has_pre_decoder_layernorm) {
            gpt_weights_.pre_decoder_layernorm.gamma = get_ptr<T>(weights_[12 * layer_num_ + 0 - weight_offset]);
            gpt_weights_.pre_decoder_layernorm.beta  = get_ptr<T>(weights_[12 * layer_num_ + 1 - weight_offset]);
        }
        else {
            weight_offset += 2;
        }
        if (gpt_variant_params_.has_post_decoder_layernorm) {
            gpt_weights_.post_decoder_layernorm.gamma = get_ptr<T>(weights_[12 * layer_num_ + 2 - weight_offset]);
            gpt_weights_.post_decoder_layernorm.beta  = get_ptr<T>(weights_[12 * layer_num_ + 3 - weight_offset]);
        }
        else {
            weight_offset += 2;
        }
        if (gpt_variant_params_.has_positional_encoding) {
            gpt_weights_.position_encoding_table = get_ptr<T>(weights_[12 * layer_num_ + 4 - weight_offset]);
            gpt_weights_.setMaxSeqLen(weights_[12 * layer_num_ + 4 - weight_offset].size(0));
        }
        else {
            weight_offset += 1;
        }
        gpt_weights_.pre_decoder_embedding_table   = get_ptr<T>(weights_[12 * layer_num_ + 5 - weight_offset]);
        gpt_weights_.post_decoder_embedding.kernel = get_ptr<T>(weights_[12 * layer_num_ + 6 - weight_offset]);

        weight_offset = 7 - weight_offset;

        for (int i = 0; i < (int)layer_num_; i++) {
            if (std::find(moe_layer_index.begin(), moe_layer_index.end(), i) != moe_layer_index.end()) {
                gpt_weights_.decoder_layer_weights[i]->ffn_weights.gating_weight.kernel =
                    get_ptr<T>(weights_[12 * layer_num_ + weight_offset + i]);
            }
        }

        weight_offset += layer_num_;

        if (gpt_variant_params_.has_adapters) {
            for (int i = 0; i < (int)layer_num_; i++) {
                gpt_weights_.decoder_layer_weights[i]->after_attention_adapter_weights.intermediate_weight.kernel =
                    get_ptr<T>(weights_[12 * layer_num_ + weight_offset + i]);
                gpt_weights_.decoder_layer_weights[i]->after_attention_adapter_weights.intermediate_weight.bias =
                    get_ptr<T>(weights_[13 * layer_num_ + weight_offset + i]);
                gpt_weights_.decoder_layer_weights[i]->after_attention_adapter_weights.output_weight.kernel =
                    get_ptr<T>(weights_[14 * layer_num_ + weight_offset + i]);
                gpt_weights_.decoder_layer_weights[i]->after_attention_adapter_weights.output_weight.bias =
                    get_ptr<T>(weights_[15 * layer_num_ + weight_offset + i]);
                gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.intermediate_weight.kernel =
                    get_ptr<T>(weights_[16 * layer_num_ + weight_offset + i]);
                gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.intermediate_weight.bias =
                    get_ptr<T>(weights_[17 * layer_num_ + weight_offset + i]);
                gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.output_weight.kernel =
                    get_ptr<T>(weights_[18 * layer_num_ + weight_offset + i]);
                gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.output_weight.bias =
                    get_ptr<T>(weights_[19 * layer_num_ + weight_offset + i]);

                if (int8_mode_ != 0) {
                    gpt_weights_.decoder_layer_weights[i]
                        ->after_attention_adapter_weights.intermediate_weight.int8_kernel =
                        get_ptr<int8_t>(int8_weights_[i + 4 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->after_attention_adapter_weights.output_weight.int8_kernel =
                        get_ptr<int8_t>(int8_weights_[i + 5 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.intermediate_weight.int8_kernel =
                        get_ptr<int8_t>(int8_weights_[i + 6 * layer_num_]);
                    gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.output_weight.int8_kernel =
                        get_ptr<int8_t>(int8_weights_[i + 7 * layer_num_]);

                    if (int8_mode == 1) {
                        gpt_weights_.decoder_layer_weights[i]
                            ->after_attention_adapter_weights.intermediate_weight.weight_only_quant_scale =
                            get_ptr<T>(scale_[i + 4 * layer_num_]);
                        gpt_weights_.decoder_layer_weights[i]
                            ->after_attention_adapter_weights.output_weight.weight_only_quant_scale =
                            get_ptr<T>(scale_[i + 5 * layer_num_]);
                        gpt_weights_.decoder_layer_weights[i]
                            ->after_ffn_adapter_weights.intermediate_weight.weight_only_quant_scale =
                            get_ptr<T>(scale_[i + 6 * layer_num_]);
                        gpt_weights_.decoder_layer_weights[i]
                            ->after_ffn_adapter_weights.output_weight.weight_only_quant_scale =
                            get_ptr<T>(scale_[i + 7 * layer_num_]);
                    }
                    else {
                        gpt_weights_.decoder_layer_weights[i]
                            ->after_attention_adapter_weights.intermediate_weight.scale =
                            get_ptr<float>(scale_[i + 4 * layer_num_]);
                        gpt_weights_.decoder_layer_weights[i]->after_attention_adapter_weights.output_weight.scale =
                            get_ptr<float>(scale_[i + 5 * layer_num_]);
                        gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.intermediate_weight.scale =
                            get_ptr<float>(scale_[i + 6 * layer_num_]);
                        gpt_weights_.decoder_layer_weights[i]->after_ffn_adapter_weights.output_weight.scale =
                            get_ptr<float>(scale_[i + 7 * layer_num_]);
                    }
                }
            }
        }

        GetConfig();
    }

    ~FTGpt() override
    {
        ft::ftNcclParamDestroy(tensor_para_);
        ft::ftNcclParamDestroy(pipeline_para_);
        ft::ftNcclParamDestroy(cache_stream_para_);
        printf("After deleting comms\n");
        cublasLtDestroy(cublasltHandle_);
        printf("After cublasLtDestroy\n");

        delete cublas_algo_map_;
        delete cublas_wrapper_mutex_;
    }

    void cleanup() override
    {
        gpt_ptr->reset();
    }

    void reset() override
    {
#ifdef TEST_FAILURES
        printf("INSIDE RESET\n");
        if (!reset_after_failure_ && gpt_ptr != NULL) {
            nccl_monitor_thread_.join();
            printf("NCCL JOINED\n");

            gpt_ptr->reset();
            printf("AFTER gpt_ptr DELETION\n");

            gpt_ptr = NULL;
            printf("gpt_ptr DELETED\n");

            reset_after_failure_ = true;
        }
#endif
    }

    void GetConfig()
    {
#if defined(SEPERATE_PROMPT) || defined(MICROBATCH_INJECTION) || defined(TEST_FAILURES)
        printf("INITIALIZE CONTROLLER CLIENT!\n");
        const char* controller_ip = std::getenv("DEJAVU_CONTROLLER_IP");
        std::string controller_address(controller_ip);
        controller_address += ":1234";

        char ip_address_[INET_ADDRSTRLEN];
        controller_client_ =
            new ControllerClient(grpc::CreateChannel(controller_address, grpc::InsecureChannelCredentials()));

        // get IP address
        int          fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
        struct ifreq ifr {};
        strcpy(ifr.ifr_name, "wlo1");
        ioctl(fd, SIOCGIFADDR, &ifr);
        close(fd);

        if (is_restart_) {
            controller_client_->IsRestart(cache_stream_para_.rank_);
        }

        ftNcclBarrier(cache_stream_para_, 0);

        strcpy(ip_address_, inet_ntoa(((sockaddr_in*)&ifr.ifr_addr)->sin_addr));
        controller_config_ = controller_client_->GetConfigInfo(cache_stream_para_.rank_);
        printf("CONTROLLER CONNECTED!\n");

        ftNcclBarrier(cache_stream_para_, 0);

        if (cache_stream_para_.rank_ == cache_stream_para_.world_size_ - 1) {
            controller_client_->ControllerReset();
        }

        ftNcclBarrier(cache_stream_para_, 0);

        printf("BARRIERS PASSED!\n");

#endif
    }

    void forward(th::Tensor&               input_ids,
                 th::Tensor&               input_lengths,
                 th::Tensor&               output_ids,
                 th::Tensor&               sequence_lengths,
                 th::Tensor&               cum_log_probs,
                 const size_t              request_output_len,
                 const size_t              beam_width,
                 th::optional<th::Tensor>  reload,
                 th::optional<th::Tensor>  streaming,
                 th::optional<th::Tensor>  swapping,
                 th::optional<th::Tensor>  top_k_opt,
                 th::optional<th::Tensor>  top_p_opt,
                 th::optional<th::Tensor>  beam_search_diversity_rate_opt,
                 th::optional<th::Tensor>  temperature_opt,
                 th::optional<th::Tensor>  len_penalty_opt,
                 th::optional<th::Tensor>  repetition_penalty_opt,
                 th::optional<th::Tensor>  presence_penalty_opt,
                 th::optional<th::Tensor>  min_length_opt,
                 th::optional<th::Tensor>  random_seed_opt,
                 th::optional<th::Tensor>  bad_words_list_opt,
                 th::optional<th::Tensor>& finished_opt,
                 th::optional<th::Tensor>  ubatch_output_lengths,
                 th::optional<th::Tensor>  ubatch_ids,
                 th::optional<int64_t>     return_cum_log_probs_opt) override
    {

        printf("Inside ParallelGptOP FORWARD!!!\n");

        int return_cum_log_probs = return_cum_log_probs_opt.has_value() ? (int)return_cum_log_probs_opt.value() : 0;
        // auto stream                 = at::cuda::getCurrentCUDAStream().stream();
        if (stream_ == NULL) {
            stream_ = (cudaStream_t*)malloc(sizeof(cudaStream_t));
            cudaStreamCreate(stream_);
            allocator = new ft::Allocator<ft::AllocatorType::TH>();
        }
        cudaStream_t   stream       = *stream_;
        cublasHandle_t cublasHandle = at::cuda::getCurrentCUDABlasHandle();
        cublasSetStream(cublasHandle, stream);
        allocator->setStream(stream);

        ft::cublasMMWrapper cublas_wrapper = ft::cublasMMWrapper(
            cublasHandle, cublasltHandle_, stream, cublas_algo_map_, cublas_wrapper_mutex_, allocator);

        if (std::is_same<T, half>::value) {
            cublas_wrapper.setGemmConfig(CUDA_R_16F, CUDA_R_16F, CUDA_R_16F, CUDA_R_32F);
        }
#ifdef ENABLE_BF16
        else if (std::is_same<T, __nv_bfloat16>::value) {
            cublas_wrapper.setBF16GemmConfig();
        }
#endif
        else if (std::is_same<T, float>::value) {
            cublas_wrapper.setFP32GemmConfig();
        }

        const size_t request_batch_size = (size_t)input_ids.size(0);
        const size_t max_input_length   = (size_t)input_ids.size(1);
        const int    total_output_len   = (int)(max_input_length + request_output_len);

        ft::AttentionType attention_type =
            ft::getAttentionType<T>(size_per_head_,
                                    ft::getSMVersion(),
                                    true,
                                    max_input_length,  // gpt supports any-seq-length fmha
                                    true,              // is_fuse
                                    false,             // with_relative_position_bias
                                    true);             // causal_mask

        if (gpt_ptr == NULL) {
#ifdef MICROBENCHMARKS
            gpt_ptr = new ft::ParallelGptDVBenchmark<T>(request_batch_size,
                                                        total_output_len,
                                                        max_input_length,
                                                        beam_width,
                                                        head_num_,
                                                        size_per_head_,
                                                        inter_size_,
                                                        layer_num_,
                                                        expert_num_,
                                                        moe_k_,
                                                        moe_layer_index_,
                                                        vocab_size_,
                                                        start_id_,
                                                        end_id_,
                                                        end_id_ + 1,  // p/prompt tuning virtual token start id
                                                        ft::PromptLearningType::no_prompt,
                                                        gpt_variant_params_,
                                                        0.0f,  // beam_search_diversity_rate,
                                                        1,     // top_k,
                                                        0.0,   // top_p,
                                                        0,     // random_seed,
                                                        1.0f,  // temperature,
                                                        1.0f,  // len_penalty,
                                                        1.0f,  // repetition_penalty,
                                                        tensor_para_,
                                                        pipeline_para_,
                                                        cache_stream_para_,
                                                        stream,
                                                        &cublas_wrapper,
                                                        allocator,
                                                        false,
                                                        &prop_,
                                                        attention_type,
                                                        false,
                                                        int8_mode_,
                                                        nullptr,
                                                        0,
                                                        shared_contexts_ratio_);

#elif MICROBATCH_INJECTION
            gpt_ptr                     = new ft::ParallelGptDVBaseline<T>(request_batch_size,
                                                       total_output_len,
                                                       max_input_length,
                                                       beam_width,
                                                       head_num_,
                                                       size_per_head_,
                                                       inter_size_,
                                                       layer_num_,
                                                       expert_num_,
                                                       moe_k_,
                                                       moe_layer_index_,
                                                       vocab_size_,
                                                       start_id_,
                                                       end_id_,
                                                       end_id_ + 1,  // p/prompt tuning virtual token start id
                                                       ft::PromptLearningType::no_prompt,
                                                       gpt_variant_params_,
                                                       0.0f,  // beam_search_diversity_rate,
                                                       1,     // top_k,
                                                       0.0,   // top_p,
                                                       0,     // random_seed,
                                                       1.0f,  // temperature,
                                                       1.0f,  // len_penalty,
                                                       1.0f,  // repetition_penalty,
                                                       tensor_para_,
                                                       pipeline_para_,
                                                       cache_stream_para_,
                                                       stream,
                                                       &cublas_wrapper,
                                                       allocator,
                                                       false,
                                                       &prop_,
                                                       attention_type,
                                                       false,
                                                       int8_mode_,
                                                       nullptr,
                                                       0,
                                                       shared_contexts_ratio_);
            gpt_ptr->controller_client_ = controller_client_;
#elif defined(TEST_FAILURES) || defined(SEPERATE_PROMPT)
            gpt_ptr                           = new ft::ParallelGptDVFT<T>(request_batch_size,
                                                 total_output_len,
                                                 max_input_length,
                                                 beam_width,
                                                 head_num_,
                                                 size_per_head_,
                                                 inter_size_,
                                                 layer_num_,
                                                 expert_num_,
                                                 moe_k_,
                                                 moe_layer_index_,
                                                 vocab_size_,
                                                 start_id_,
                                                 end_id_,
                                                 end_id_ + 1,  // p/prompt tuning virtual token start id
                                                 ft::PromptLearningType::no_prompt,
                                                 gpt_variant_params_,
                                                 0.0f,  // beam_search_diversity_rate,
                                                 1,     // top_k,
                                                 0.0,   // top_p,
                                                 0,     // random_seed,
                                                 1.0f,  // temperature,
                                                 1.0f,  // len_penalty,
                                                 1.0f,  // repetition_penalty,
                                                 tensor_para_,
                                                 pipeline_para_,
                                                 cache_stream_para_,
                                                 prompt_world_size_,
                                                 token_world_size_,
                                                 stream,
                                                 &cublas_wrapper,
                                                 allocator,
                                                 false,
                                                 &prop_,
                                                 attention_type,
                                                 false,
                                                 int8_mode_,
                                                 nullptr,
                                                 0,
                                                 shared_contexts_ratio_);
            gpt_ptr->controller_client_       = controller_client_;
            gpt_ptr->start_config_ubatch_ids_ = std::vector<int>(controller_config_.ubatch_global_ids().begin(),
                                                                 controller_config_.ubatch_global_ids().end());
            gpt_ptr->start_config_steps_ =
                std::vector<int>(controller_config_.ubatch_steps().begin(), controller_config_.ubatch_steps().end());
            gpt_ptr->prompt_seen_global_ids_ =
                std::set<int>(controller_config_.prompts_seen().begin(), controller_config_.prompts_seen().end());
            gpt_ptr->start_has_failed_        = controller_config_.has_failed();
            gpt_ptr->start_stream_cache_next_ = controller_config_.stream_cache_next();
            gpt_ptr->start_stream_cache_prev_ = controller_config_.stream_cache_prev();
            printf("BOOLEANS: %d, %d, %d\n", gpt_ptr->start_has_failed_, gpt_ptr->start_stream_cache_next_,  gpt_ptr->start_stream_cache_prev_);
            gpt_ptr->reset_ = false;
#else
            gpt_ptr = new ft::ParallelGpt<T>(request_batch_size,
                                             total_output_len,
                                             max_input_length,
                                             beam_width,
                                             head_num_,
                                             size_per_head_,
                                             inter_size_,
                                             layer_num_,
                                             expert_num_,
                                             moe_k_,
                                             moe_layer_index_,
                                             vocab_size_,
                                             start_id_,
                                             end_id_,
                                             end_id_ + 1,  // p/prompt tuning virtual token start id
                                             ft::PromptLearningType::no_prompt,
                                             gpt_variant_params_,
                                             0.0f,  // beam_search_diversity_rate,
                                             1,     // top_k,
                                             0.0,   // top_p,
                                             0,     // random_seed,
                                             1.0f,  // temperature,
                                             1.0f,  // len_penalty,
                                             1.0f,  // repetition_penalty,
                                             tensor_para_,
                                             cache_stream_para_,
                                             stream,
                                             &cublas_wrapper,
                                             allocator,
                                             false,
                                             &prop_,
                                             attention_type,
                                             false,
                                             int8_mode_,
                                             nullptr,
                                             0,
                                             shared_contexts_ratio_);
#endif
        }

        std::vector<uint32_t> output_seq_len(request_batch_size, total_output_len);
        if (ubatch_output_lengths.has_value()) {
            for (int i = 0; i < request_batch_size; i++)
                output_seq_len[i] = ubatch_output_lengths.value()[i].item<int>() + max_input_length;
        }
        std::unordered_map<std::string, ft::Tensor> input_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"input_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, max_input_length},
                        get_ptr<int>(input_ids)}},
            {"input_lengths",
             ft::Tensor{
                 ft::MEMORY_GPU, ft::TYPE_INT32, std::vector<size_t>{request_batch_size}, get_ptr<int>(input_lengths)}},
            {"output_seq_len",
             ft::Tensor{
                 ft::MEMORY_CPU, ft::TYPE_UINT32, std::vector<size_t>{request_batch_size}, output_seq_len.data()}}};

        if (beam_width > 1 && beam_search_diversity_rate_opt.has_value()) {
            input_tensors.insert(
                {"beam_search_diversity_rate",
                 convert_tensor<float>(beam_search_diversity_rate_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (top_p_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_p", convert_tensor<float>(top_p_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (top_k_opt.has_value()) {
            input_tensors.insert(
                {"runtime_top_k", convert_tensor<uint>(top_k_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (temperature_opt.has_value()) {
            input_tensors.insert(
                {"temperature", convert_tensor<float>(temperature_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (len_penalty_opt.has_value()) {
            input_tensors.insert(
                {"len_penalty", convert_tensor<float>(len_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (repetition_penalty_opt.has_value()) {
            input_tensors.insert({"repetition_penalty",
                                  convert_tensor<float>(repetition_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (presence_penalty_opt.has_value()) {
            input_tensors.insert(
                {"presence_penalty", convert_tensor<float>(presence_penalty_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (min_length_opt.has_value()) {
            input_tensors.insert(
                {"min_length", convert_tensor<int>(min_length_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (random_seed_opt.has_value()) {
            input_tensors.insert(
                {"random_seed",
                 convert_tensor<unsigned long long int>(random_seed_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }

        if (bad_words_list_opt.has_value()) {
            CHECK_INPUT(bad_words_list_opt.value(), torch::kInt32);
            input_tensors.insert({"bad_words_list", convert_tensor<int>(bad_words_list_opt.value())});
        }

        bool return_context_cum_log_probs = false;
        if (return_cum_log_probs == 2) {
            return_context_cum_log_probs = true;
            input_tensors.insert(
                {"is_return_context_cum_log_probs",
                 ft::Tensor{ft::MEMORY_CPU, ft::TYPE_BOOL, std::vector<size_t>{1}, &return_context_cum_log_probs}});
        }

        if (ubatch_ids.has_value()) {
            input_tensors.insert(
                {"ubatch_given_ids", convert_tensor<int>(ubatch_ids.value(), ft::MemoryType::MEMORY_CPU)});
        }

        std::unordered_map<std::string, ft::Tensor> output_tensors = std::unordered_map<std::string, ft::Tensor>{
            {"output_ids",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width, (size_t)total_output_len},
                        get_ptr<int>(output_ids)}},
            {"sequence_length",
             ft::Tensor{ft::MEMORY_GPU,
                        ft::TYPE_INT32,
                        std::vector<size_t>{request_batch_size, beam_width},
                        get_ptr<int>(sequence_lengths)}}};

        if (return_cum_log_probs > 0) {
            output_tensors.insert({"cum_log_probs",
                                   ft::Tensor{ft::MEMORY_GPU,
                                              ft::TYPE_FP32,
                                              std::vector<size_t>{request_batch_size, beam_width},
                                              get_ptr<float>(cum_log_probs)}});
        }

        if (finished_opt.has_value()) {
            input_tensors.insert({"finished", convert_tensor<bool>(finished_opt.value(), ft::MemoryType::MEMORY_CPU)});
        }

        if (reload.has_value()) {
            input_tensors.insert({"reload", convert_tensor<bool>(reload.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (streaming.has_value()) {
            input_tensors.insert({"streaming", convert_tensor<bool>(streaming.value(), ft::MemoryType::MEMORY_CPU)});
        }
        if (swapping.has_value()) {
            input_tensors.insert({"swapping", convert_tensor<bool>(swapping.value(), ft::MemoryType::MEMORY_CPU)});
        }


        std::vector<ft::Tensor> ret;
        try {
#ifdef TEST_FAILURES
            if (!gpt_ptr->reset_) {
                gpt_ptr->comp_thread_ = std::thread(
                    &ft::ParallelGptDVFT<T>::thread_forward, gpt_ptr, &output_tensors, &input_tensors, &gpt_weights_);
                gpt_ptr->comp_done_ = false;
                if (gpt_ptr->global_iteration_ == 0) {
                    nccl_monitor_thread_ = std::thread(&ft::ParallelGptDVFT<T>::monitor_nccl, gpt_ptr);
                    printf("AFTER THREAD CREATION!\n");
                }
                gpt_ptr->comp_thread_.join();
                printf("COMP THREAD JOINED!\n");
            }
            //nccl_monitor_thread_.join();
            if (gpt_ptr->teptr_ != nullptr) {
                printf("GOT EXCEPTION!\n");
                std::exception_ptr exp_ptr = std::move(gpt_ptr->teptr_);
                reset();
                printf("After ftGPT RESET!\n");
                std::rethrow_exception(exp_ptr);
            }
#else
            gpt_ptr->forward(&output_tensors, &input_tensors, &gpt_weights_);
#endif
        }
        catch (const std::runtime_error& error) {
            FT_LOG_ERROR(error.what());
            printf("------------------ GOT AN ERROR! RETURN!\n");
            ft::FT_CHECK(false);
        }
        catch (const std::exception& error) {
            // printf("GOT AN ERROR!\n");
            FT_LOG_ERROR(error.what());
            ft::FT_CHECK(false);
        }
        catch (...) {
            FT_LOG_ERROR("Unknown error");
            ft::FT_CHECK(false);
        }
    }

private:
    const int64_t        head_num_;
    const int64_t        size_per_head_;
    const int64_t        inter_size_;
    const int64_t        layer_num_;
    const int64_t        expert_num_;
    const int64_t        moe_k_;
    std::vector<int64_t> moe_layer_index_;
    const int64_t        vocab_size_;
    const int64_t        start_id_;
    const int64_t        end_id_;
    const double         shared_contexts_ratio_;
    const int64_t        prompt_world_size_;
    const int64_t        token_world_size_;
    const int64_t        torch_rank_;
    const bool           is_restart_;

    const int64_t int8_mode_ = 0;
    bool reset_after_failure_ = false;

    int64_t tensor_para_size_;
    int64_t pipeline_para_size_;
#if defined(SEPERATE_PROMPT) || defined(MICROBATCH_INJECTION) || defined(TEST_FAILURES)
    ControllerClient*   controller_client_ = NULL;
    StartUpInfoResponse controller_config_;
#endif

    ft::gptVariantParams gpt_variant_params_;

    std::vector<th::Tensor> int8_weights_;
    std::vector<th::Tensor> scale_;
    std::vector<th::Tensor> weights_;

    ft::NcclParam tensor_para_;
    ft::NcclParam pipeline_para_;
    ft::NcclParam cache_stream_para_;

    cublasLtHandle_t         cublasltHandle_;
    std::mutex*              cublas_wrapper_mutex_;
    ft::cublasAlgoMap*       cublas_algo_map_;
    struct cudaDeviceProp    prop_;
    ft::ParallelGptWeight<T> gpt_weights_;
    int                      world_size_ = 1;
    int                      rank_       = 0;
    std::thread              nccl_monitor_thread_;
};

class Reset: public th::jit::CustomClassHolder {
    public:
        Reset();
};

class ParallelGptOp: public th::jit::CustomClassHolder {
public:
    ParallelGptOp(const int64_t              head_num,
                  const int64_t              size_per_head,
                  const int64_t              inter_size,
                  const int64_t              layer_num,
                  const int64_t              expert_num,
                  const int64_t              moe_k,
                  const std::vector<int64_t> moe_layer_index,
                  const int64_t              vocab_size,
                  const int64_t              start_id,
                  const int64_t              end_id,
                  const int64_t              tensor_para_size,
                  const int64_t              pipeline_para_size,
                  const int64_t              int8_mode,
                  const double               layernorm_eps,
                  const std::string          layernorm_type,
                  const std::string          activation_type,
                  const bool                 has_positional_encoding,
                  const bool                 has_pre_decoder_layernorm,
                  const bool                 has_post_decoder_layernorm,
                  const bool                 has_adapters,
                  const int64_t              adapter_inter_size,
                  const bool                 use_attention_linear_bias,
                  const vector<th::Tensor>   weights,
                  const vector<th::Tensor>   int8_weights,
                  const vector<th::Tensor>   scale,
                  const double               shared_contexts_ratio,
                  const int64_t              prompt_world_size,
                  const int64_t              token_world_size,
                  const int64_t              torch_rank,
                  const bool                 is_restart);

    ~ParallelGptOp();

    vector<th::Tensor> forward(th::Tensor               input_ids,
                               th::Tensor               input_lengths,
                               const int64_t            output_len,
                               th::optional<th::Tensor> reload,
                               th::optional<th::Tensor> streaming,
                               th::optional<th::Tensor> swapping,
                               th::optional<int64_t>    beam_width_opt,
                               th::optional<th::Tensor> top_k_opt,
                               th::optional<th::Tensor> top_p_opt,
                               th::optional<th::Tensor> beam_search_diversity_rate_opt,
                               th::optional<th::Tensor> temperature_opt,
                               th::optional<th::Tensor> len_penalty_opt,
                               th::optional<th::Tensor> repetition_penalty_opt,
                               th::optional<th::Tensor> presence_penalty_opt,
                               th::optional<th::Tensor> min_length_opt,
                               th::optional<th::Tensor> random_seed_opt,
                               th::optional<th::Tensor> bad_words_list_opt,
                               th::optional<th::Tensor> finished_opt,
                               th::optional<th::Tensor> ubatch_output_lengths,
                               th::optional<th::Tensor> ubatch_ids,
                               th::optional<int64_t>    return_cum_log_probs_opt);

    void cleanup();
    void reset();


private:
    const at::ScalarType    st_;
    IFGpt*                  ftgpt;
    std::vector<th::Tensor> weights;
};


}  // namespace torch_ext
