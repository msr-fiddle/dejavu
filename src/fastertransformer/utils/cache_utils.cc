#include "src/fastertransformer/utils/cache_utils.h"
#include <cstddef>

namespace fastertransformer {

BaseCacheManager::BaseCacheManager(PipelineConfig      prompt_config,
                                   PipelineConfig      token_config,
                                   int                 my_rank,
                                   std::vector<void*>& key_cache,
                                   std::vector<void*>& value_cache,
                                   std::vector<void*>& host_cache_addr,
                                   int                 prompt_buffer_slots,
                                   int                 fp_size,
                                   size_t              ubatch_size,
                                   size_t              local_num_layer,
                                   size_t              prompt_size,
                                   size_t              max_seq_len,
                                   size_t              local_hidden_units,
                                   size_t              local_head_num,
                                   size_t              size_per_head,
                                   int                 beam_width,
                                   cudaStream_t        fetch_key_copy_stream,
                                   cudaStream_t        fetch_value_copy_stream,
                                   cudaStream_t        flush_key_copy_stream,
                                   cudaStream_t        flush_value_copy_stream):
    prompt_config_(prompt_config),
    token_config_(token_config),
    my_rank_(my_rank),
    key_cache_(key_cache),
    value_cache_(value_cache),
    host_cache_addr_(host_cache_addr),
    prompt_buffer_slots_(prompt_buffer_slots),
    fp_size_(fp_size),
    ubatch_size_(ubatch_size),
    local_num_layer_(local_num_layer),
    prompt_size_(prompt_size),
    max_seq_len_(max_seq_len),
    local_hidden_units_(local_hidden_units),
    local_head_num_(local_head_num),
    size_per_head_(size_per_head),
    beam_width_(beam_width),
    fetch_key_copy_stream_(fetch_key_copy_stream),
    fetch_value_copy_stream_(fetch_value_copy_stream),
    flush_key_copy_stream_(flush_key_copy_stream),
    flush_value_copy_stream_(flush_value_copy_stream)

{

    cache_size_       = local_num_layer * ubatch_size_ * beam_width_ * max_seq_len_ * local_hidden_units_ * fp_size_;
    per_layer_offset_ = ubatch_size_ * beam_width_ * max_seq_len_ * local_hidden_units_ * fp_size_;
    kc_size_          = 16 / fp_size_;
    vc_size_          = size_per_head_;

    size_t per_layer_size = local_hidden_units_ * ubatch_size_ * beam_width_ * fp_size_ * prompt_size_;
    CUDACHECK(cudaMalloc(&prompt_key_temp_addr_, per_layer_size));
    CUDACHECK(cudaMalloc(&prompt_value_temp_addr_, per_layer_size));

    // for a single token
    size_t per_token_size = local_num_layer * local_hidden_units * fp_size_ * ubatch_size_ * beam_width_;
    CUDACHECK(cudaMalloc(&token_key_temp_addr_, per_token_size));
    CUDACHECK(cudaMalloc(&token_value_temp_addr_, per_token_size));

    // for 'transfer_size_' tokens
    size_t new_seq_size =
        local_num_layer * local_hidden_units_ * ubatch_size_ * beam_width_ * fp_size_ * transfer_size_;
    CUDACHECK(cudaMalloc(&buff_key_, new_seq_size));
    CUDACHECK(cudaMalloc(&buff_value_, new_seq_size));
}

BaseCacheManager::~BaseCacheManager()
{
    CUDACHECK(cudaFree(prompt_key_temp_addr_));
    CUDACHECK(cudaFree(prompt_value_temp_addr_));
    CUDACHECK(cudaFree(token_key_temp_addr_));
    CUDACHECK(cudaFree(token_value_temp_addr_));
    CUDACHECK(cudaFree(buff_key_));
    CUDACHECK(cudaFree(buff_value_));
}

void BaseCacheManager::flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream)
{
}

void BaseCacheManager::fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream)
{
}

void BaseCacheManager::gather(StreamInfo args)
{

    size_t seq_size               = args.seq_end - args.seq_start;
    size_t per_layer_batch_offset = local_hidden_units_ * fp_size_ * beam_width_ * max_seq_len_;
    size_t per_layer_size         = local_hidden_units_ * args.batch_size * beam_width_ * fp_size_ * seq_size;

    size_t key_scaling_factor =
        local_num_layer_ * args.batch_size * beam_width_ * local_head_num_ * size_per_head_ / (16 / fp_size_);
    size_t value_scaling_factor = local_num_layer_ * args.batch_size * beam_width_ * local_head_num_;

    if (args.is_prompt) {

        for (auto layer : args.layer_ids) {

            // printf("[IN GATHER], Process layer %d, ubatch id is %d, dst address is %p, src is %p\n", layer,
            // args.batch_idx, args.dst_key_start, key_cache_[args.batch_idx]);

            size_t layer_ubatch_offset =
                layer * per_layer_offset_;  //+ args.batch_idx * args.batch_size * per_layer_batch_offset;

            char* dst = NULL;
            if (args.dst_key_start != NULL) {
                dst = (char*)(args.dst_key_start) + layer * per_layer_size;
            }
            fetch(prompt_key_temp_addr_, per_layer_size, args.peer_rank, dst, fetch_key_copy_stream_);

            invoke_transfer_kernel_prompt(prompt_key_temp_addr_,
                                          (char*)(key_cache_[args.batch_idx]) + layer_ubatch_offset,
                                          local_head_num_ * size_per_head_ / (16 / fp_size_) * beam_width_
                                              * args.batch_size,
                                          seq_size,
                                          kc_size_ * seq_size * fp_size_,
                                          kc_size_ * max_seq_len_ * fp_size_,
                                          kc_size_,
                                          fp_size_,
                                          fetch_key_copy_stream_);

            dst = NULL;
            if (args.dst_value_start != NULL) {
                dst = (char*)(args.dst_value_start) + layer * per_layer_size;
            }
            fetch(prompt_value_temp_addr_, per_layer_size, args.peer_rank, dst, fetch_value_copy_stream_);

            invoke_transfer_kernel_prompt(prompt_value_temp_addr_,
                                          (char*)(value_cache_[args.batch_idx]) + layer_ubatch_offset,
                                          local_head_num_ * beam_width_ * args.batch_size,
                                          seq_size,
                                          vc_size_ * seq_size * fp_size_,
                                          vc_size_ * max_seq_len_ * fp_size_,
                                          vc_size_,
                                          fp_size_,
                                          fetch_value_copy_stream_);
        }
    }

    else {
        size_t key_row_size   = max_seq_len_ * kc_size_ * fp_size_;
        size_t value_row_size = max_seq_len_ * vc_size_ * fp_size_;
        size_t per_token_size = local_num_layer_ * local_hidden_units_ * fp_size_ * ubatch_size_ * beam_width_;
        per_layer_size        = local_hidden_units_ * args.batch_size * beam_width_ * fp_size_ * args.seq_start;

        if (args.seq_end < prompt_size_)
            return;

        // A balance between memory and transfer overheads: cut cache into chunks
        int num_chunks = (args.seq_end - args.seq_start) / transfer_size_;
        if ((args.seq_end - args.seq_start) % transfer_size_ != 0)
            num_chunks += 1;

        for (int i = 0; i < num_chunks; i++) {
            int chunk_start = args.seq_start + i * transfer_size_;
            int chunk_end   = chunk_start + transfer_size_;
            if (i == num_chunks - 1)
                chunk_end = args.seq_end;

            size_t new_seq_size = local_num_layer_ * local_hidden_units_ * ubatch_size_ * beam_width_ * fp_size_
                                  * (chunk_end - chunk_start);
            char* dst = NULL;
            if (args.dst_key_start != NULL) {
                dst = (char*)(args.dst_key_start) + local_num_layer_ * per_layer_size;
            }
            //printf("IN GATHER, fetch %d bytes!\n", new_seq_size);
            fetch(buff_key_, new_seq_size, args.peer_rank, dst, fetch_key_copy_stream_);
            invoke_multi_token_transfer((char*)buff_key_,
                                        (void*)(key_cache_[args.batch_idx]),
                                        kc_size_,
                                        kc_size_ * fp_size_,
                                        key_scaling_factor,
                                        chunk_start,
                                        chunk_end - chunk_start,
                                        key_row_size,
                                        per_token_size,
                                        fp_size_,
                                        fetch_key_copy_stream_);

            dst = NULL;
            if (args.dst_value_start != NULL) {
                dst = (char*)(args.dst_value_start) + local_num_layer_ * per_layer_size;
            }
            size_t buff_size =
                local_num_layer_ * local_hidden_units_ * ubatch_size_ * beam_width_ * fp_size_ * transfer_size_;

            // printf(
            //     "FETCH VALUE, BATCH IDX IS %d, seq len is %d, value_row_size %u, chunk_start %u, value_scaling_factor %u, vc_size_ %u, per_token_size %u, new_seq_size %u, BUFF SIZE %u\n",
            //     args.batch_idx,
            //     chunk_end - chunk_start,
            //     value_row_size,
            //     chunk_start,
            //     value_scaling_factor,
            //     vc_size_,
            //     per_token_size,
            //     new_seq_size,
            //     buff_size);
            fetch(buff_value_, new_seq_size, args.peer_rank, dst, fetch_value_copy_stream_);
            //CUDACHECK(cudaDeviceSynchronize());

            invoke_multi_token_transfer((char*)buff_value_,
                                        (void*)(value_cache_[args.batch_idx]),
                                        vc_size_,
                                        vc_size_ * fp_size_,
                                        value_scaling_factor,
                                        chunk_start,
                                        chunk_end - chunk_start,
                                        value_row_size,
                                        per_token_size,
                                        fp_size_,
                                        fetch_value_copy_stream_);
            //CUDACHECK(cudaDeviceSynchronize());
        }
    }
}

void BaseCacheManager::scatter(StreamInfo args)
{

    size_t seq_size               = args.seq_end - args.seq_start;
    size_t per_layer_batch_offset = local_hidden_units_ * fp_size_ * beam_width_ * max_seq_len_;
    size_t per_layer_size         = local_hidden_units_ * args.batch_size * beam_width_ * fp_size_ * seq_size;

    size_t key_scaling_factor =
        local_num_layer_ * args.batch_size * beam_width_ * local_head_num_ * size_per_head_ / (16 / fp_size_);
    size_t value_scaling_factor = local_num_layer_ * args.batch_size * beam_width_ * local_head_num_;

    // assumes cache is in GPU
    // TODO: we might need to reconsider this distriction
    if (args.is_prompt) {

        for (auto layer : args.layer_ids) {

            size_t layer_ubatch_offset = layer * per_layer_offset_;

            // printf("[IN SCATTER], Process layer %d, ubatch id is %d, dst address is %p, src is %p\n", layer,
            //  args.batch_idx, args.dst_key_start, key_cache_[args.batch_idx]);

            // key cache
            invoke_transfer_kernel_prompt((char*)(key_cache_[args.batch_idx]) + layer_ubatch_offset,
                                          prompt_key_temp_addr_,
                                          local_head_num_ * beam_width_ * size_per_head_ / (16 / fp_size_)
                                              * args.batch_size,
                                          seq_size,
                                          kc_size_ * max_seq_len_ * fp_size_,
                                          kc_size_ * seq_size * fp_size_,
                                          kc_size_,
                                          fp_size_,
                                          flush_key_copy_stream_);

            char* dst = NULL;
            if (args.dst_key_start != NULL) {
                dst = (char*)(args.dst_key_start) + layer * per_layer_size;
            }
            flush(prompt_key_temp_addr_, per_layer_size, args.peer_rank, dst, flush_value_copy_stream_);

            // value cache
            invoke_transfer_kernel_prompt((char*)(value_cache_[args.batch_idx]) + layer_ubatch_offset,
                                          prompt_value_temp_addr_,
                                          local_head_num_ * beam_width_ * args.batch_size,
                                          seq_size,
                                          vc_size_ * max_seq_len_ * fp_size_,
                                          vc_size_ * seq_size * fp_size_,
                                          vc_size_,
                                          fp_size_,
                                          flush_value_copy_stream_);
            dst = NULL;
            if (args.dst_value_start != NULL) {
                // TODO: What should 'dst_value_start' be?
                dst = (char*)(args.dst_value_start) + layer * per_layer_size;
            }
            flush(prompt_value_temp_addr_, per_layer_size, args.peer_rank, dst, flush_value_copy_stream_);
        }
    }
    else {

        size_t per_token_size        = local_num_layer_ * local_hidden_units_ * fp_size_ * ubatch_size_ * beam_width_;
        size_t per_layer_prompt_size = local_hidden_units_ * args.batch_size * beam_width_ * fp_size_ * prompt_size_;
        size_t key_row_size          = max_seq_len_ * kc_size_ * fp_size_;
        size_t value_row_size        = max_seq_len_ * vc_size_ * fp_size_;

        // printf(
        //     "Batch idx is %d, Key scaling factor is %d, key row size is %d, token_cache_size is %d, prompt_offset
        //     is %d, prompt_size_ is %d, step is %d\n", args.batch_idx, key_scaling_factor, key_row_size,
        //     per_token_size,
        //     local_num_layer_ * per_layer_prompt_size,
        //     prompt_size_,
        //     args.seq_start);

        // key cache
        invoke_single_token_transfer((void*)(key_cache_[args.batch_idx]),
                                     token_key_temp_addr_,
                                     kc_size_,
                                     kc_size_ * fp_size_,
                                     key_scaling_factor,
                                     args.seq_start,
                                     key_row_size,
                                     per_token_size,
                                     fp_size_,
                                     flush_key_copy_stream_);
        char* dst = NULL;
        if (args.dst_key_start != NULL) {
            dst = (char*)(args.dst_key_start) + local_num_layer_ * per_layer_prompt_size
                  + (args.seq_start - prompt_size_) * per_token_size;
        }

        flush(token_key_temp_addr_, per_token_size, args.peer_rank, dst, flush_key_copy_stream_);

        // value cache
        dst = NULL;
        invoke_single_token_transfer(value_cache_[args.batch_idx],
                                     token_value_temp_addr_,
                                     vc_size_,
                                     vc_size_ * fp_size_,
                                     value_scaling_factor,
                                     args.seq_start,
                                     value_row_size,
                                     per_token_size,
                                     fp_size_,
                                     flush_value_copy_stream_);
        if (args.dst_value_start != NULL) {
            dst = (char*)(args.dst_value_start) + local_num_layer_ * per_layer_prompt_size
                  + (args.seq_start - prompt_size_) * per_token_size;
        }
        flush(token_value_temp_addr_, per_token_size, args.peer_rank, dst, flush_value_copy_stream_);
    }
}

void BaseCacheManager::stream_in(StreamInfo args)
{
    if (!args.is_prompt || prompt_config_.is_same(token_config_)) {
        // TODO: check this
        args.peer_rank = my_rank_ - token_config_.pp_depth_ * token_config_.tp_size_;
        gather(args);
    }
    else if (prompt_config_.pp_depth_ != token_config_.pp_depth_) {
        // that's token calling
        int pp_layers_per_stage = (local_num_layer_ * token_config_.pp_depth_) / prompt_config_.pp_depth_;
        std::vector<std::vector<int>> pp_ranks(prompt_config_.pp_depth_, std::vector<int>{});
        for (auto l : args.layer_ids) {
            int dst_rank = l / pp_layers_per_stage;
            pp_ranks[dst_rank].push_back(l);
        }

        for (int i = 0; i < prompt_config_.pp_depth_; i++) {
            if (!pp_ranks[i].empty()) {
                StreamInfo rank_args = {pp_ranks[i],
                                        args.batch_idx,
                                        args.batch_size,
                                        args.seq_start,
                                        args.seq_end,
                                        args.is_prompt,
                                        i,
                                        // TODO: not sure about these
                                        args.dst_key_start,
                                        args.dst_value_start};
                gather(rank_args);
            }
        }
    }
}

void BaseCacheManager::stream_out(StreamInfo args)
{
    if (!args.is_prompt || prompt_config_.is_same(token_config_)) {
        // TODO: check this
        args.peer_rank = token_config_.pp_depth_ * token_config_.tp_size_ + my_rank_;
        scatter(args);
    }

    else if (prompt_config_.pp_depth_ != token_config_.pp_depth_) {

        int tp_layers_per_stage = (local_num_layer_ * prompt_config_.pp_depth_) / token_config_.pp_depth_;
        std::vector<std::vector<int>> tp_ranks(token_config_.pp_depth_, std::vector<int>{});
        for (auto l : args.layer_ids) {
            int dst_rank = l / tp_layers_per_stage + token_config_.pp_depth_ * token_config_.tp_size_;
            tp_ranks[dst_rank].push_back(l);
        }
        for (int i = 0; i < token_config_.pp_depth_; i++) {
            if (!tp_ranks[i].empty()) {
                // same batch size
                StreamInfo rank_args = {tp_ranks[i],
                                        args.batch_idx,
                                        args.batch_size,
                                        args.seq_start,
                                        args.seq_end,
                                        args.is_prompt,
                                        i + token_config_.pp_depth_ * token_config_.tp_size_,
                                        // TODO: not sure about these
                                        args.dst_key_start,
                                        args.dst_value_start};
                scatter(rank_args);
            }
        }
    }

    // TODO: what to do when different batch sizes are used?
}

void BaseCacheManager::sync_streams()
{
    CUDACHECK(cudaStreamSynchronize(fetch_key_copy_stream_));
    CUDACHECK(cudaStreamSynchronize(fetch_value_copy_stream_));
    CUDACHECK(cudaStreamSynchronize(flush_key_copy_stream_));
    CUDACHECK(cudaStreamSynchronize(flush_value_copy_stream_));
}

void BaseCacheManager::baseline_scatter(StreamInfo args)
{

    // just calls 'flush' on contiguous regions

    size_t key_scaling_factor   = args.batch_size * beam_width_ * local_head_num_ * size_per_head_ / (16 / fp_size_);
    size_t value_scaling_factor = args.batch_size * beam_width_ * local_head_num_;

    size_t num_key_transfers   = local_num_layer_ * key_scaling_factor;
    size_t num_value_transfers = local_num_layer_ * value_scaling_factor;
    char*  dst                 = NULL;
    printf("IN BaselineCacheManager::scatter, num_key_transfers %d, key_scaling_factor %d\n",
           num_key_transfers,
           key_scaling_factor);

    if (args.is_prompt) {

        size_t prompt_size = args.seq_end - args.seq_start;

        for (auto layer : args.layer_ids) {
            printf("IN BaselineCacheManager::scatter, layer %d\n", layer);

            size_t layer_ubatch_offset = layer * per_layer_offset_;

            // key
            char* layer_src = (char*)(key_cache_[args.batch_idx]) + layer_ubatch_offset;
            for (int i = 0; i < key_scaling_factor; i++) {
                char* new_src = layer_src + max_seq_len_ * kc_size_ * fp_size_;
                if (args.dst_key_start != NULL) {
                    dst = (char*)(args.dst_key_start) + layer_ubatch_offset + max_seq_len_ * kc_size_ * fp_size_;
                }

                // copy locally
                flush(new_src, prompt_size * kc_size_ * fp_size_, args.peer_rank, dst, flush_key_copy_stream_);

                // copy remotely
                flush(dst, prompt_size * kc_size_ * fp_size_, args.peer_rank, NULL, flush_key_copy_stream_);
            }

            dst = NULL;
            // value
            layer_src = (char*)(value_cache_[args.batch_idx]) + layer_ubatch_offset;
            for (int i = 0; i < value_scaling_factor; i++) {
                char* new_src = layer_src + max_seq_len_ * vc_size_ * fp_size_;
                if (args.dst_value_start != NULL) {
                    dst = (char*)(args.dst_value_start) + layer_ubatch_offset + max_seq_len_ * vc_size_ * fp_size_;
                }
                flush(new_src, prompt_size * vc_size_ * fp_size_, args.peer_rank, dst, flush_value_copy_stream_);
                flush(dst, prompt_size * vc_size_ * fp_size_, args.peer_rank, NULL, flush_value_copy_stream_);
            }
        }
    }
    else {

        // key
        for (int i = 0; i < num_key_transfers; i++) {
            // printf("%d, %d, %d, %d\n", i, num_key_transfers, args.seq_start, max_seq_len_);
            char* new_src = (char*)(key_cache_[args.batch_idx]) + i * max_seq_len_ * kc_size_ * fp_size_
                            + args.seq_start * kc_size_ * fp_size_;
            if (args.dst_key_start != NULL) {
                dst = (char*)(args.dst_key_start) + i * max_seq_len_ * kc_size_ * fp_size_
                      + args.seq_start * kc_size_ * fp_size_;
            }
            flush(new_src, kc_size_ * fp_size_, args.peer_rank, dst, flush_key_copy_stream_);
            flush(dst, kc_size_ * fp_size_, args.peer_rank, NULL, flush_key_copy_stream_);
        }

        // value
        for (int i = 0; i < num_value_transfers; i++) {
            char* new_src = (char*)(key_cache_[args.batch_idx]) + i * max_seq_len_ * vc_size_ * fp_size_
                            + args.seq_start * vc_size_ * fp_size_;
            if (args.dst_value_start != NULL) {
                dst = (char*)(args.dst_value_start) + i * max_seq_len_ * vc_size_ * fp_size_
                      + args.seq_start * vc_size_ * fp_size_;
            }
            flush(new_src, vc_size_ * fp_size_, args.peer_rank, dst, flush_value_copy_stream_);
            flush(dst, vc_size_ * fp_size_, args.peer_rank, NULL, flush_value_copy_stream_);
        }
    }
}

void BaseCacheManager::baseline_gather(StreamInfo args)
{

    size_t key_scaling_factor   = args.batch_size * beam_width_ * local_head_num_ * size_per_head_ / (16 / fp_size_);
    size_t value_scaling_factor = args.batch_size * beam_width_ * local_head_num_;

    size_t num_key_transfers   = local_num_layer_ * key_scaling_factor;
    size_t num_value_transfers = local_num_layer_ * value_scaling_factor;
    char*  dst                 = NULL;
    printf("IN BaselineCacheManager::gather, num_key_transfers %d, key_scaling_factor %d\n",
           num_key_transfers,
           key_scaling_factor);

    if (args.is_prompt) {

        size_t prompt_size = args.seq_end - args.seq_start;

        for (auto layer : args.layer_ids) {
            size_t layer_ubatch_offset = layer * per_layer_offset_;
            // key
            for (int i = 0; i < key_scaling_factor; i++) {
                if (args.dst_key_start != NULL) {
                    dst = (char*)(args.dst_key_start) + layer_ubatch_offset + max_seq_len_ * kc_size_ * fp_size_;
                }
                fetch(dst, prompt_size * kc_size_ * fp_size_, args.peer_rank, NULL, flush_key_copy_stream_);
            }

            dst = NULL;
            // value
            for (int i = 0; i < value_scaling_factor; i++) {
                if (args.dst_value_start != NULL) {
                    dst = (char*)(args.dst_value_start) + layer_ubatch_offset + max_seq_len_ * vc_size_ * fp_size_;
                }
                fetch(dst, prompt_size * vc_size_ * fp_size_, args.peer_rank, NULL, flush_value_copy_stream_);
            }
        }
    }
    else {
        // key
        for (int i = 0; i < num_key_transfers; i++) {
            // printf("%d, %d, %d, %d\n", i, num_key_transfers, args.seq_start, max_seq_len_);
            if (args.dst_key_start != NULL) {
                dst = (char*)(args.dst_key_start) + i * max_seq_len_ * kc_size_ * fp_size_
                      + args.seq_start * kc_size_ * fp_size_;
            }
            fetch(dst, kc_size_ * fp_size_, args.peer_rank, NULL, flush_key_copy_stream_);
        }

        // value
        for (int i = 0; i < num_value_transfers; i++) {
            if (args.dst_value_start != NULL) {
                dst = (char*)(args.dst_value_start) + i * max_seq_len_ * vc_size_ * fp_size_
                      + args.seq_start * vc_size_ * fp_size_;
            }
            fetch(dst, vc_size_ * fp_size_, args.peer_rank, NULL, flush_value_copy_stream_);
        }
    }
}

void NCCLCacheManager::flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream)
{
    // printf("At NCCL FLUSH: source %p, dest rank %d, size %lu\n", src_address, dst_rank, num_bytes);
    NCCLCHECK(ncclSend(src_address, num_bytes, ncclUint8, dst_rank, comm_, stream));
}

void NCCLCacheManager::fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream)
{
    // printf("At NCCL FETCH: dst %p, src rank %d, size %lu\n", dst_address, src_rank, num_bytes);
    NCCLCHECK(ncclRecv(dst_address, num_bytes, ncclUint8, src_rank, comm_, stream));
}

void MPICacheManager::flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream)
{
    if (dst_address != NULL) {
        CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyDeviceToHost, stream));
    }
    else {

        // printf("At MPI FLUSH: src %p, dst rank %d, size %lu, NT_MAX is %lu\n", src_address, dst_rank, num_bytes,
        // (size_t)INT_MAX); MPICHECK(MPI_Send(src_address, num_bytes, MPI_CHAR, dst_rank, 0, MPI_COMM_WORLD));

        size_t transfers = num_bytes / (size_t)INT_MAX;
        if ((num_bytes % (size_t)INT_MAX) != 0)
            transfers += 1;

        CUDACHECK(cudaStreamSynchronize(stream));

        printf("At MPI FLUSH: src %p, dst rank %d, size %lu, transfers is %lu, INT_MAX is %lu\n",
               src_address,
               dst_rank,
               num_bytes,
               transfers,
               (size_t)INT_MAX);

        for (int i = 0; i < transfers; i++) {

            int bytes_to_transfer = (i == transfers - 1) ? num_bytes : INT_MAX;

            MPICHECK(MPI_Send(src_address, bytes_to_transfer, MPI_CHAR, dst_rank, 0, MPI_COMM_WORLD));
            num_bytes -= (size_t)INT_MAX;
        }
    }
}

void MPICacheManager::fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream)
{
    if (src_address != NULL) {
        CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyHostToDevice, stream));
    }
    else {

        MPI_Status status;
        // printf("At MPI RECV: src %p, src rank %d, size %lu\n", src_address, src_rank, num_bytes);
        // MPICHECK(MPI_Recv(dst_address, num_bytes, MPI_CHAR, src_rank, 0, MPI_COMM_WORLD, &status));

        size_t transfers = num_bytes / (size_t)INT_MAX;
        if ((num_bytes % (size_t)INT_MAX) != 0)
            transfers += 1;

        CUDACHECK(cudaStreamSynchronize(stream));

        for (int i = 0; i < transfers; i++) {

            int bytes_to_transfer = (i == transfers - 1) ? num_bytes : INT_MAX;

            // printf("At MPI RECV: src %p, dst rank %d, size %lu\n", src_address, dst_rank, bytes_to_transfer);
            MPICHECK(MPI_Recv(dst_address, bytes_to_transfer, MPI_CHAR, src_rank, 0, MPI_COMM_WORLD, &status));
            num_bytes -= (size_t)INT_MAX;
        }
    }
}

void MPIRMACacheManager::flush(
    void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream)
{
    if (dst_address != NULL) {
        CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyDeviceToHost, stream));
    }
    else {
        CUDACHECK(cudaStreamSynchronize(stream));
        MPI_Aint target_disp = (char*)src_address - (char*)host_cache_addr_[0];
        // printf("--------------------------- SRC IS %p, TARGET_DISP IS %d, PROMPT_SLOT_ is %d, write %lu bytes\n",
        //        src_address,
        //        target_disp,
        //        prompt_slot_,
        //        num_bytes);

        MPI_Put(src_address, num_bytes, MPI_CHAR, dst_rank, target_disp, num_bytes, MPI_CHAR, rma_win_);
    }
}

void MPIRMACacheManager::fetch(
    void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream)
{
    assert(src_address != NULL);
    CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyHostToDevice, stream));
}

void TCPCacheManager::flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream)
{

    if (dst_address != NULL) {
        // printf("AT TCP FLUSH, LOCAL! SRC_ADDRESS IS %p, DST ADDRESS IS %p, SIZE IS %lu\n", src_address, dst_address,
        // num_bytes);
        CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyDeviceToHost, stream));
    }
    else {
        ip::tcp::socket*          socket = (*sockets_)[dst_rank];
        boost::system::error_code ec;

        printf("AT TCP FLUSH! SOCKET ADDR IS %p, dst_rank is %d, send %d bytes\n", socket, dst_rank, num_bytes);
        size_t bytes_written = write(
            *socket, buffer(src_address, num_bytes), transfer_exactly(num_bytes), ec);  // we need to write num_bytes
        if (ec) {
            printf("BOOST ERROR OCCURED WHILE WRITING!\n");
            return;
        }
        assert(bytes_written == num_bytes);
    }
}

void TCPCacheManager::fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream)
{
}

void LocalCacheManager::flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream)
{
    // printf("At flush: source %p, dest %p, size %lu\n", src_address, dst_address, num_bytes);
    CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyDeviceToHost, stream));
}

void LocalCacheManager::fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream)
{
    CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyHostToDevice, stream));
}

void BaselineCacheManager::flush(
    void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream)
{
    if (dst_address != NULL) {
        CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyDeviceToHost, stream));
    }
    else {
        // printf("At MPI FLUSH: src %p, dst rank %d, size %lu\n", src_address, dst_rank, num_bytes);

        CUDACHECK(cudaStreamSynchronize(stream));
        MPICHECK(MPI_Send(src_address, num_bytes, MPI_CHAR, dst_rank, 0, MPI_COMM_WORLD));
    }
}

void BaselineCacheManager::fetch(
    void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream)
{
    if (src_address != NULL) {
        CUDACHECK(cudaMemcpyAsync(dst_address, src_address, num_bytes, cudaMemcpyHostToDevice, stream));
    }
    else {
        // printf("At MPI FETCH: dst %p, src rank %d, size %lu\n", dst_address, src_rank, num_bytes);

        CUDACHECK(cudaStreamSynchronize(stream));
        MPI_Status status;
        MPICHECK(MPI_Recv(dst_address, num_bytes, MPI_CHAR, src_rank, 0, MPI_COMM_WORLD, &status));
    }
}

}  // namespace fastertransformer
