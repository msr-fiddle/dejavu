/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include <atomic>
#include <vector>

#include "src/fastertransformer/kernels/add_residual_kernels.h"
#include "src/fastertransformer/kernels/layernorm_kernels.h"
#include "src/fastertransformer/layers/BaseLayer.h"
#include "src/fastertransformer/layers/TensorParallelGeluFfnLayer.h"
#include "src/fastertransformer/layers/TensorParallelReluFfnLayer.h"
#include "src/fastertransformer/layers/attention_layers/TensorParallelGptContextAttentionLayer.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ParallelGptDecoderLayerWeight.h"
#include "src/fastertransformer/utils/Tensor.h"
#include "src/fastertransformer/utils/allocator.h"
#include "src/fastertransformer/utils/cublasMMWrapper.h"
#include "src/fastertransformer/utils/custom_ar_comm.h"

#include "src/fastertransformer/models/multi_gpu_gpt/StateStreamServer.h"
#include "src/fastertransformer/utils/cache_utils.h"

namespace fastertransformer {

template<typename T>
class ParallelGptContextDecoder: public BaseLayer {
private:
    void*              key_ubatch_address_   = nullptr;
    void*              value_ubatch_address_ = nullptr;
    BaseCacheManager** cache_manager_        = nullptr;
    std::vector<bool>* is_token_phase_       = nullptr;

    std::vector<bool>          stream_threads_created_;
    std::atomic_bool**         layer_ready_;
    int                        layers_per_pp_;
    int                        num_microbatches_;
    int                        layers_per_tp_;
    int                        tp_per_pp_;
    std::vector<DejaVuClient*> dejavu_clients_;

    // buffer handling
    size_t max_batch_size_ = 0;
    size_t max_seq_len_    = 0;
    int    beam_width_     = 0;
    int    prompt_len_     = 0;
    int    num_slots_      = 0;

    // meta data
    size_t               head_num_;
    size_t               size_per_head_;
    size_t               inter_size_;
    size_t               num_layer_;
    size_t               num_valid_layer_;
    size_t               expert_num_;
    size_t               moe_k_;
    std::vector<int64_t> moe_layer_index_;
    float                layernorm_eps_;
    LayerNormType        layernorm_type_;
    ActivationType       activation_type_;

    // adapter
    bool   has_adapters_;
    size_t adapter_inter_size_;
    T*     after_adapter_attn_output_;

    // calculated data
    size_t hidden_units_;

    NcclParam  tensor_para_;
    NcclParam  pipeline_para_;
    NcclParam* cache_stream_para_;

    std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm_;
    int                                 enable_custom_all_reduce_;
    AttentionType                       attention_type_;

    // quantization
    int int8_mode_ = 0;
    // NOTE (perkzz): dynamic_quant enabled
    bool   dynamic_quant_                  = true;
    float* attention_query_dynamic_scale_  = nullptr;
    float* ffn_intermediate_dynamic_scale_ = nullptr;

    bool is_qk_buf_float_;

    BaseAttentionLayer<T>* self_attention_layer_;
    FfnLayer<T>*           ffn_layer_;

    void allocateBuffer() override;
    void allocateBuffer(size_t batch_size, size_t seq_len, bool use_shared_contexts);
    void freeBuffer() override;
    bool isValidLayerParallelId(uint l);
    void initialize();
    bool isFirstLayerParallelId(uint l);
    bool isLastLayerParallelId(uint l);
    int  getFirstLayerParallelId();
    void gpu_sync(cudaStream_t stream);
    void copy_kv_cache_ubatch_layer(
        int layer, int microbatch_id, int prompt_size, int microbatch_size, size_t total_cache_size);
    void copy_thread_func(int ite, int prompt_size, int microbatch_size, size_t total_cache_size);
    void flush_layer(int l, void* start_addr, size_t total_cache_size);

    T*      decoder_normed_input_    = nullptr;
    T*      self_attn_output_        = nullptr;
    T*      normed_self_attn_output_ = nullptr;
    T*      decoder_layer_output_    = nullptr;
    size_t* h_pinned_token_num_ptr_  = nullptr;
    int*    padding_offset_          = nullptr;
    int*    cu_seqlens_              = nullptr;

    T*   compact_decoder_features_ = nullptr;
    T*   compact_attention_mask_   = nullptr;
    int* compact_input_lengths_    = nullptr;
    T*   k_cache_layer_            = nullptr;
    T*   v_cache_layer_            = nullptr;

    T*   expert_scales_                            = nullptr;
    int* expanded_source_row_to_expanded_dest_row_ = nullptr;
    int* expert_for_source_row_                    = nullptr;
    T*   fc2_result_                               = nullptr;
    T*   adapter_fc2_result_                       = nullptr;

    // for copying
    std::vector<void*>* host_address_         = nullptr;
    void**              key_prompt_address_   = nullptr;
    void**              value_prompt_address_ = nullptr;

    bool   copy_cache_ = false;
    size_t memory_len_;
    size_t kc_offset_;
    size_t vc_offset_;
    size_t per_layer_offset_;
    size_t key_scaling_factor_;
    size_t value_scaling_factor_;
    size_t layer_prompt_size_;
    size_t cache_size_;
    size_t ubatch_layer_prompt_size_;
    size_t per_layer_ubatch_offset_;

protected:
public:
    std::atomic_bool*        restart;
    std::vector<std::thread> stream_threads_;
    std::atomic_bool         thread_done_;

    ParallelGptContextDecoder(size_t                              max_batch_size,
                              size_t                              max_seq_len,
                              size_t                              head_num,
                              size_t                              size_per_head,
                              size_t                              inter_size,
                              size_t                              num_layer,
                              size_t                              expert_num,
                              size_t                              moe_k,
                              std::vector<int64_t>                moe_layer_index,
                              float                               layernorm_eps,
                              gptVariantParams                    gpt_variant_params,
                              NcclParam                           tensor_para,
                              NcclParam                           pipeline_para,
                              cudaStream_t                        stream,
                              cublasMMWrapper*                    cublas_wrapper,
                              IAllocator*                         allocator,
                              bool                                is_free_buffer_after_forward,
                              bool                                is_qk_buf_float,
                              std::vector<void*>*                 host_address,
                              AttentionType                       attention_type           = AttentionType::UNFUSED_MHA,
                              bool                                sparse                   = false,
                              int                                 int8_mode                = 0,
                              std::shared_ptr<AbstractCustomComm> custom_all_reduce_comm   = nullptr,
                              int                                 enable_custom_all_reduce = 0,
                              void**                              key_prompt_address       = nullptr,
                              void**                              value_prompt_address     = nullptr,
                              NcclParam*                          cache_stream_para        = nullptr,
                              bool                                copy_cache               = false,
                              BaseCacheManager**                  cache_manager            = nullptr,
                              std::vector<bool>*                  is_token_phase           = nullptr,
                              int                                 num_slots                = 0);

    ParallelGptContextDecoder(ParallelGptContextDecoder<T> const& decoder);

    ~ParallelGptContextDecoder();

    void forward(TensorMap*                                            output_tensors,
                 const TensorMap*                                      input_tensors,
                 const std::vector<ParallelGptDecoderLayerWeight<T>*>* decoder_layer_weights);

    void set_cache_info(int beam_width, int max_input_len);

    void set_streaming_info(int                        num_micobatches,
                            int                        layers_per_tp,
                            int                        tp_per_pp,
                            std::vector<DejaVuClient*> dejavu_clients);
};

}  // namespace fastertransformer
