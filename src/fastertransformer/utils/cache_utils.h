#include <algorithm>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
// #include <arpa/inet.h>
// #include <linux/if.h>
#include <boost/asio.hpp>
#ifdef BUILD_MULTI_GPU
#include <mpi.h>
#include <nccl.h>
#endif

#include "src/fastertransformer/kernels/decoding_kernels.h"

#define CUDACHECK(cmd)                                                                                                 \
    do {                                                                                                               \
        cudaError_t e = cmd;                                                                                           \
        if (e != cudaSuccess) {                                                                                        \
            printf("Failed: Cuda error %s:%d '%s'\n", __FILE__, __LINE__, cudaGetErrorString(e));                      \
            throw std::runtime_error("CUDA FAILURE - THROW ERROR!");                                                   \
        }                                                                                                              \
    } while (0)

#ifdef BUILD_MULTI_GPU
#define MPICHECK(cmd)                                                                                                  \
    do {                                                                                                               \
        int e = cmd;                                                                                                   \
        if (e != MPI_SUCCESS) {                                                                                        \
            printf("Failed: MPI error %s:%d '%d'\n", __FILE__, __LINE__, e);                                           \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)
#else
#define MPICHECK(cmd) printf("[WARNING] No MPI\n");
#endif

#define NCCLCHECK(cmd)                                                                                                 \
    do {                                                                                                               \
        ncclResult_t r = cmd;                                                                                          \
        if (r != ncclSuccess) {                                                                                        \
            printf("Failed, NCCL error %s:%d '%s'\n", __FILE__, __LINE__, ncclGetErrorString(r));                      \
            exit(EXIT_FAILURE);                                                                                        \
        }                                                                                                              \
    } while (0)

using namespace boost::asio;
namespace fastertransformer {

typedef struct CacheEntry {
    bool   reserved;
    bool   written;
    bool   seen;
    size_t timestamp_res;
} CacheEntry;

typedef struct StreamInfo {
    std::vector<int> layer_ids;
    int              batch_idx;  // e.g. if it is a specific microbatch
    int              batch_size;
    int              seq_start;
    int              seq_end;
    bool             is_prompt;  // TODO: we might not need this
    int              peer_rank;
    void*            dst_key_start;
    void*            dst_value_start;
} StreamInfo;

typedef struct PipelineConfig {

    int pp_depth_;
    int tp_size_;
    int batch_size_;
    int microbatch_size_;

    bool is_same(struct PipelineConfig other)
    {
        return (pp_depth_ == other.pp_depth_ && tp_size_ == other.tp_size_ && batch_size_ == other.batch_size_
                && microbatch_size_ == other.microbatch_size_);
    }

} PipelineConfig;

class BaseCacheManager {

public:
    PipelineConfig     prompt_config_;
    PipelineConfig     token_config_;
    std::vector<void*> host_cache_addr_;

    int prompt_slot_ = 0;
    int prompt_buffer_slots_;

    int    my_rank_;
    int    fp_size_;
    size_t ubatch_size_;
    size_t local_num_layer_;
    size_t prompt_size_;
    size_t max_seq_len_;
    size_t local_hidden_units_;
    size_t local_head_num_;
    size_t size_per_head_;
    int    beam_width_;

    std::vector<void*> key_cache_;    // in GPU
    std::vector<void*> value_cache_;  // in GPU

    cudaStream_t fetch_key_copy_stream_;
    cudaStream_t fetch_value_copy_stream_;

    cudaStream_t flush_key_copy_stream_;
    cudaStream_t flush_value_copy_stream_;

    ncclComm_t comm_;

    size_t cache_size_;
    size_t per_layer_offset_;
    size_t vc_size_;
    size_t kc_size_;
    size_t transfer_size_ = 100;  // can adjust accordingly

    void* prompt_key_temp_addr_;
    void* prompt_value_temp_addr_;
    void* token_key_temp_addr_;
    void* token_value_temp_addr_;
    void* buff_key_;
    void* buff_value_;

    BaseCacheManager(PipelineConfig      prompt_config,
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
                     cudaStream_t        flush_value_copy_stream);

    ~BaseCacheManager();

    virtual void flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream);
    virtual void fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream);

    void baseline_gather(StreamInfo args);
    void baseline_scatter(StreamInfo args);

    // TODO: ADD 2 STREAMS
    void gather(StreamInfo args);
    void scatter(StreamInfo args);

    void scatter_in(StreamInfo args);
    void scatter_out(StreamInfo args);

    void sync_streams();
};

class NCCLCacheManager: public BaseCacheManager {

private:
    ncclComm_t comm_;

public:
    NCCLCacheManager(PipelineConfig      prompt_config,
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
                     cudaStream_t        flush_value_copy_stream,
                     ncclComm_t          comm):

        BaseCacheManager{prompt_config,
                         token_config,
                         my_rank,
                         key_cache,
                         value_cache,
                         host_cache_addr,
                         prompt_buffer_slots,
                         fp_size,
                         ubatch_size,
                         local_num_layer,
                         prompt_size,
                         max_seq_len,
                         local_hidden_units,
                         local_head_num,
                         size_per_head,
                         beam_width,
                         fetch_key_copy_stream,
                         fetch_value_copy_stream,
                         flush_key_copy_stream,
                         flush_value_copy_stream}
    {
        comm_ = comm;
    }

    void flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream);
    void fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream);
};

class MPICacheManager: public BaseCacheManager {
public:
    MPICacheManager(PipelineConfig      prompt_config,
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
                    cudaStream_t        flush_key_copy_stream   = 0,
                    cudaStream_t        flush_value_copy_stream = 0):

        BaseCacheManager{prompt_config,
                         token_config,
                         my_rank,
                         key_cache,
                         value_cache,
                         host_cache_addr,
                         prompt_buffer_slots,
                         fp_size,
                         ubatch_size,
                         local_num_layer,
                         prompt_size,
                         max_seq_len,
                         local_hidden_units,
                         local_head_num,
                         size_per_head,
                         beam_width,
                         fetch_key_copy_stream,
                         fetch_value_copy_stream,
                         flush_key_copy_stream,
                         flush_value_copy_stream}
    {
    }

    void flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream);
    void fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream);
};

class MPIRMACacheManager: public BaseCacheManager {
private:
    MPI_Win rma_win_;

public:
    MPIRMACacheManager(PipelineConfig      prompt_config,
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
                       cudaStream_t        flush_key_copy_stream   = 0,
                       cudaStream_t        flush_value_copy_stream = 0):
        BaseCacheManager{prompt_config,
                         token_config,
                         my_rank,
                         key_cache,
                         value_cache,
                         host_cache_addr,
                         prompt_buffer_slots,
                         fp_size,
                         ubatch_size,
                         local_num_layer,
                         prompt_size,
                         max_seq_len,
                         local_hidden_units,
                         local_head_num,
                         size_per_head,
                         beam_width,
                         fetch_key_copy_stream,
                         fetch_value_copy_stream,
                         flush_key_copy_stream,
                         flush_value_copy_stream}
    {

        // 2*cache_size for key and value cache. The starting point is the key cache pointer

        size_t rma_size = host_cache_addr_.size() * cache_size_ * 2;
        MPICHECK(MPI_Win_create(
            host_cache_addr_[0], prompt_buffer_slots * rma_size, 1, MPI_INFO_NULL, MPI_COMM_WORLD, &rma_win_));
        MPICHECK(MPI_Win_fence(0, rma_win_));
    }

    ~MPIRMACacheManager()
    {
        // MPICHECK(MPI_Win_free(&rma_win_));
    }

    void flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream);
    void fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream);
};

class TCPCacheManager: public BaseCacheManager {
private:
    std::vector<ip::tcp::socket*>* sockets_;

public:
    TCPCacheManager(PipelineConfig                 prompt_config,
                    PipelineConfig                 token_config,
                    int                            my_rank,
                    std::vector<void*>&            key_cache,
                    std::vector<void*>&            value_cache,
                    std::vector<void*>&            host_cache_addr,
                    int                            prompt_buffer_slots,
                    int                            fp_size,
                    size_t                         ubatch_size,
                    size_t                         local_num_layer,
                    size_t                         prompt_size,
                    size_t                         max_seq_len,
                    size_t                         local_hidden_units,
                    size_t                         local_head_num,
                    size_t                         size_per_head,
                    int                            beam_width,
                    std::vector<ip::tcp::socket*>* sockets,
                    cudaStream_t                   fetch_key_copy_stream,
                    cudaStream_t                   fetch_value_copy_stream,
                    cudaStream_t                   flush_key_copy_stream   = 0,
                    cudaStream_t                   flush_value_copy_stream = 0):
        BaseCacheManager{prompt_config,
                         token_config,
                         my_rank,
                         key_cache,
                         value_cache,
                         host_cache_addr,
                         prompt_buffer_slots,
                         fp_size,
                         ubatch_size,
                         local_num_layer,
                         prompt_size,
                         max_seq_len,
                         local_hidden_units,
                         local_head_num,
                         size_per_head,
                         beam_width,
                         fetch_key_copy_stream,
                         fetch_value_copy_stream,
                         flush_key_copy_stream,
                         flush_value_copy_stream}
    {
        sockets_ = sockets;
    }

    ~TCPCacheManager() {}

    void flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream);
    void fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream);
};

class LocalCacheManager: public BaseCacheManager {
public:
    LocalCacheManager(PipelineConfig      prompt_config,
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
                      cudaStream_t        flush_key_copy_stream   = 0,
                      cudaStream_t        flush_value_copy_stream = 0):
        BaseCacheManager{prompt_config,
                         token_config,
                         my_rank,
                         key_cache,
                         value_cache,
                         host_cache_addr,
                         prompt_buffer_slots,
                         fp_size,
                         ubatch_size,
                         local_num_layer,
                         prompt_size,
                         max_seq_len,
                         local_hidden_units,
                         local_head_num,
                         size_per_head,
                         beam_width,
                         fetch_key_copy_stream,
                         fetch_value_copy_stream,
                         flush_key_copy_stream,
                         flush_value_copy_stream}
    {
    }

    void flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream);
    void fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream);
};

class BaselineCacheManager: public BaseCacheManager {
public:
    BaselineCacheManager(PipelineConfig      prompt_config,
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
                         cudaStream_t        flush_key_copy_stream   = 0,
                         cudaStream_t        flush_value_copy_stream = 0):

        BaseCacheManager{prompt_config,
                         token_config,
                         my_rank,
                         key_cache,
                         value_cache,
                         host_cache_addr,
                         prompt_buffer_slots,
                         fp_size,
                         ubatch_size,
                         local_num_layer,
                         prompt_size,
                         max_seq_len,
                         local_hidden_units,
                         local_head_num,
                         size_per_head,
                         beam_width,
                         fetch_key_copy_stream,
                         fetch_value_copy_stream,
                         flush_key_copy_stream,
                         flush_value_copy_stream}
    {
    }

    // MPI for now
    void flush(void* src_address, size_t num_bytes, int dst_rank, void* dst_address, cudaStream_t stream);
    void fetch(void* dst_address, size_t num_bytes, int src_rank, void* src_address, cudaStream_t stream);
};

}  // namespace fastertransformer
