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

#pragma once

#include <cstddef>
#include <queue>
#include <vector>

#include <arpa/inet.h>
#include <atomic>
#include <fcntl.h>
#include <sys/mman.h>
#include <thread>

#include <condition_variable>
#include <mutex>

#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptContextDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoder.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptWeight.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"
// #include "src/fastertransformer/utils/cache_utils.h"

namespace fastertransformer {

static void*                         recv_addr_;
static bool                          socket_set;

static ncclComm_t*  stream_comm_ = NULL;
static ncclUniqueId stream_id_;
static int          stream_rank_;
static int          nStreamRanks_;
static int          global_iteration_ = 0;


template<typename T>
class ParallelGptDVBenchmark: public BaseLayer {
private:
    // meta data
    size_t               head_num_;
    size_t               size_per_head_;
    size_t               inter_size_;
    size_t               num_layer_;
    size_t               vocab_size_;
    size_t               expert_num_;
    size_t               moe_k_;
    std::vector<int64_t> moe_layer_index_;

    std::mutex         step_mtx_;
    std::vector<void*> mapped_host_addr_;
    std::thread        stream_thread_;
    int                prev_prompt_size_ = 0;

    int current_slot_id_;
    // 0: dummy baseline, 1: only batching, 2: batching + prompt, 3: DV
    int baseline_ = 3;

    int    start_id_;
    int    end_id_;
    float  beam_search_diversity_rate_;
    size_t hidden_units_;

    const float layernorm_eps_;  // OPT
    float       shared_contexts_ratio_;

    // TODO(bhsueh) remove these member because they are runtime parameters
    size_t             top_k_;
    float              top_p_;
    unsigned long long random_seed_;

    float temperature_;
    float len_penalty_;
    float repetition_penalty_;

    size_t    local_head_num_;
    NcclParam tensor_para_;
    NcclParam pipeline_para_;
    NcclParam cache_stream_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;

    const bool    is_context_qk_buf_float_ = (std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM") == nullptr
                                           || std::string(std::getenv("CONTEXT_ATTENTION_BMM1_HALF_ACCUM")) != "ON");
    size_t        vocab_size_padded_;
    const int     int8_mode_      = 0;
    AttentionType attention_type_ = AttentionType::UNFUSED_MHA;

    // Prompt Learning Parameters
    PromptLearningType prompt_learning_type_;
    int                prompt_learning_start_id_;  // start_id for prompt_learning (only needed by prefix prompts)
    bool               has_p_prompt_tuning_;
    bool               has_prefix_prompt_;
    bool               has_prefix_soft_prompt_;

    bool swapping_   = false;
    bool config_set_ = false;

    // GPT Variants parameters: e.g. Meta OPT
    gptVariantParams gpt_variant_params_;

    ParallelGptDecoder<T>*                  gpt_decoder_;
    ParallelGptContextDecoder<T>*           gpt_context_decoder_;
    std::vector<DynamicDecodeLayer<float>*> dynamic_decode_layer_;
    BaseCacheManager*                       cache_manager_;

    double ms_heartbeat_ = 1000.0;  // how often to send heartbeats to the controller
    char   ip_address_[INET_ADDRSTRLEN];

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size,
                        size_t beam_width,
                        size_t max_seq_len,
                        size_t memory_len,
                        size_t max_input_len,
                        bool   is_return_context_cum_log_probs,
                        bool   reload);
    void initializeVectors(const size_t num_microbatches);
    void freeBuffer() override;
    bool isValidLayerParallelId(uint l);
    void gpu_sync(cudaStream_t stream);
    void initialize();

    void copy_no_optim(bool is_prompt, size_t start_step, size_t end_step, int local_batch_size, int ubatch_id);
    void open_and_map(size_t size, bool reload);
    void do_flush(size_t start_step, size_t end_step, int ubatch_id);
    void copy_thread_func(int num_microbatches, int prompt_size, int local_batch_size);
    void copy_cache_decode(size_t step, int local_batch_size, int ubatch_id);
    void read_state(int num_microbatches, int prompt_size, int local_batch_size, int ubatch_id);
    void receive_cache_ubatch_layer(size_t prompt_size, int microbatch_id, int beam_width, int local_batch_size);
    void receive_cache_ubatch_step(size_t step, int microbatch_id, int beam_width, int local_batch_size);

    void computeContextCumLogProbs(float*                      cum_log_probs,
                                   const T*                    context_decoder_outputs,
                                   const int*                  input_ids,
                                   const int*                  input_lengths,
                                   const size_t                batch_size,
                                   const size_t                beam_width,
                                   const size_t                max_input_length,
                                   const ParallelGptWeight<T>* gpt_weights,
                                   int                         ite);

protected:
    // For stateful processing (interactive generation)
    int               step_;
    bool              cont_gen = false;
    size_t            session_len_;
    size_t            memory_len_;
    size_t            layers_per_pp_;
    std::vector<int*> tiled_total_padding_count_;

    T*       padded_embedding_kernel_;
    const T* padded_embedding_kernel_ptr_;

    std::vector<T*> tiled_input_attention_mask_;

    std::vector<T*>        decoder_input_buf_;
    std::vector<T*>        decoder_normed_input_buf_;
    std::vector<T*>        decoder_output_buf_;
    std::vector<T*>        normed_decoder_output_buf_;
    std::vector<float*>    logits_buf_;
    std::vector<float*>    nccl_logits_buf_;
    std::vector<float*>    cum_log_probs_;
    std::vector<bool*>     finished_buf_;
    std::vector<int*>      sequence_lengths_;
    std::vector<uint32_t*> seq_limit_len_;
    bool*                  microbatch_should_stop_;
    int                    num_slots_ = 2;  // for swapping

    std::vector<int*> shared_contexts_idx_;
    std::vector<T*>   compact_decoder_features_;
    std::vector<int*> compact_idx_;
    std::vector<int*> batch_to_compact_idx_;
    std::vector<int*> compact_size_;

    std::vector<T*>    key_cache_;
    std::vector<T*>    value_cache_;
    std::vector<void*> key_cache_void_;
    std::vector<void*> value_cache_void_;
    int*               cache_indirections_[2] = {nullptr, nullptr};  // TODO: not sure if this needs to change

    std::vector<int*> start_ids_buf_;  // TODO: check if these are needed
    std::vector<int*> end_ids_buf_;

    std::vector<int*> tiled_input_ids_buf_;
    std::vector<int*> tiled_input_lengths_buf_;

    std::vector<cudaEvent_t*> key_swapping_events_;
    std::vector<cudaEvent_t*> value_swapping_events_;

    // prompt_learning weight_batch ptrs
    std::vector<const T**> prompt_learning_weight_batch_;
    std::vector<int*>      tiled_prompt_lengths_buf_;  // only needed by prefix prompts

    std::vector<int*>  transposed_output_ids_buf_;
    std::vector<int*>  output_ids_buf_;
    std::vector<int*>  output_ids_buf_cpu_;
    std::vector<int*>  parent_ids_buf_;
    std::vector<bool*> tiled_masked_tokens_;

    std::vector<T*>     context_decoder_input_buf_;
    std::vector<T*>     context_decoder_normed_input_buf_;
    std::vector<T*>     context_decoder_output_buf_;
    std::vector<float*> output_log_probs_buf_;

    // The slope per head of an attention linear bias.
    T* linear_bias_slopes_;

    // buffers dedicated to log prob computation
    std::vector<T*>     lp_normed_decoder_output_buf_;
    std::vector<float*> lp_logits_buf_;
    std::vector<float*> lp_nccl_logits_buf_;
    std::vector<float*> lp_logprob_buf_;

    // for cache streaming
    // TODO: for DV Streaming, fix these!
    void* key_device_addr_;
    void* value_device_addr_;
    void* key_host_addr_;
    void* value_host_addr_;
    void* key_prompt_layer_addr_;
    void* value_prompt_layer_addr_;
    void* key_prompt_ubatch_layer_addr_;
    void* value_prompt_ubatch_layer_addr_;

    int flush_interval_ = 1;
    int kc_offset_      = 0;
    int vc_offset_      = 0;

    size_t total_cache_size_              = 0;
    size_t ubatch_cache_size_             = 0;
    size_t token_cache_size_              = 0;
    size_t key_scaling_factor_            = 0;
    size_t value_scaling_factor_          = 0;
    size_t key_scaling_factor_no_layer_   = 0;
    size_t value_scaling_factor_no_layer_ = 0;
    size_t output_ids_size_               = 0;
    size_t per_layer_offset_              = 0;
    size_t layer_prompt_size_             = 0;
    size_t layer_prompt_ubatch_size_      = 0;
    size_t layer_prompt_ubatch_offset_    = 0;

    std::vector<bool> ubatch_phase_;  // 0 for prompt, 1 for token
    std::atomic_int*  ubatch_step_;
    std::vector<int>  ubatch_step_start_;
    std::vector<int>  ubatch_step_restart_;
    std::vector<int>  ubatch_step_end_;
    std::vector<int>  ubatch_global_id_;
    std::vector<bool> done_;

    cudaStream_t fetch_key_stream_;
    cudaStream_t fetch_value_stream_;

    cudaStream_t flush_key_stream_;
    cudaStream_t flush_value_stream_;

    std::atomic_int computation_step_;
    std::atomic_int copy_step_;
    int             last_flush_step_;
    int*            last_token_;

    // function pointer callback
    using callback_sig                 = void(std::unordered_map<std::string, Tensor>*, void*);
    callback_sig* token_generated_cb_  = nullptr;
    void*         token_generated_ctx_ = nullptr;

    void setOutputTensors(std::unordered_map<std::string, Tensor>*       output_tensors,
                          const std::unordered_map<std::string, Tensor>* input_tensors,
                          const size_t                                   gen_len,
                          const size_t                                   session_len,
                          const size_t                                   max_context_len,
                          const size_t                                   max_input_without_prompt_length);
    void sendTensorsToFirstPipelineNode(std::unordered_map<std::string, Tensor>*       output_tensors,
                                        const std::unordered_map<std::string, Tensor>* input_tensors);

public:
    ParallelGptDVBenchmark(size_t               max_batch_size,
                           size_t               max_seq_len,
                           size_t               max_input_len,
                           size_t               beam_width,
                           size_t               head_num,
                           size_t               size_per_head,
                           size_t               inter_size,
                           size_t               num_layer,
                           size_t               expert_num,
                           size_t               moe_k,
                           std::vector<int64_t> moe_layer_index,
                           size_t               vocab_size,
                           int                  start_id,
                           int                  end_id,
                           int                  prompt_learning_start_id,  // only needed by p/prompt-tuning
                           PromptLearningType   prompt_learning_type,
                           gptVariantParams     gpt_variant_params,
                           float                beam_search_diversity_rate,
                           size_t               top_k,
                           float                top_p,
                           unsigned long long   random_seed,
                           float                temperature,
                           float                len_penalty,
                           float                repetition_penalty,
                           NcclParam            tensor_para,
                           NcclParam            pipeline_para,
                           NcclParam            cache_stream_para,
                           cudaStream_t         stream,
                           cublasMMWrapper*     cublas_wrapper,
                           IAllocator*          allocator,
                           bool                 is_free_buffer_after_forward,
                           cudaDeviceProp*      cuda_device_prop                        = nullptr,
                           AttentionType        attention_type                          = AttentionType::UNFUSED_MHA,
                           bool                 sparse                                  = false,
                           int                  int8_mode                               = 0,
                           std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                           int                                 enable_custom_all_reduce = 0,
                           float                               shared_contexts_ratio    = 1.0f);

    ParallelGptDVBenchmark(ParallelGptDVBenchmark<T> const& gpt);

    ~ParallelGptDVBenchmark();

    std::atomic_bool comp_done_;

    void forward(std::vector<Tensor>*        output_tensors,
                 const std::vector<Tensor>*  input_tensors,
                 const ParallelGptWeight<T>* gpt_weights);

    void forward(std::unordered_map<std::string, Tensor>*       output_tensors,
                 const std::unordered_map<std::string, Tensor>* input_tensors,
                 const ParallelGptWeight<T>*                    gpt_weights);

    size_t getPipelineParallelRank();
    size_t getPipelineParallelSize();
    size_t getTensorParallelRank();
    size_t getTensorParallelSize();
    size_t getHiddenUnits();
    size_t getStep();
    bool*  getFinishBuffer();

    void registerCallback(callback_sig* fn, void* ctx);
    void unRegisterCallback();

    void reset();

};

}  // namespace fastertransformer
