/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/utils/nccl_utils.h"

namespace fastertransformer {

#ifdef BUILD_MULTI_GPU
template<typename T>
ncclDataType_t getNcclDataType()
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, float>::value) {
        nccl_data_type = ncclFloat;
    }
    else if (std::is_same<T, half>::value) {
        nccl_data_type = ncclHalf;
    }
#if defined(ENABLE_BF16) && defined(ENABLE_BF16_NCCL)
    else if (std::is_same<T, __nv_bfloat16>::value) {
        nccl_data_type = ncclBfloat16;
    }
#endif
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else if (std::is_same<T, char>::value) {
        nccl_data_type = ncclChar;
    }
    else if (std::is_same<T, bool>::value) {
        nccl_data_type = ncclInt8;
    }
    else {
        printf("[ERROR] NCCL only support float, half, bfloat16, int, char, and bool. \n");
        exit(-1);
    }
    return nccl_data_type;
}
#endif

template<typename T>
void ftNcclAllReduceSum(const T* send_buf, T* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclAllReduce(
        (const void*)send_buf, (void*)recv_buf, data_size, nccl_data_type, ncclSum, nccl_param.nccl_comm_, stream));
    NCCLCHECK(ncclGroupEnd());
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void ftNcclAllGather(
    const T* send_buf, T* recv_buf, const int data_size, const int rank, NcclParam nccl_param, cudaStream_t stream)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(
        ncclAllGather(send_buf + rank * data_size, recv_buf, data_size, nccl_data_type, nccl_param.nccl_comm_, stream));
    NCCLCHECK(ncclGroupEnd());
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template<typename T>
void ftNcclSend(const T* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclSend(send_buf, data_size, nccl_data_type, peer, nccl_param.nccl_comm_, stream));
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template void
ftNcclSend(const float* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclSend(const half* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#ifdef ENABLE_BF16
template void ftNcclSend(
    const __nv_bfloat16* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#endif
template void
ftNcclSend(const int* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclSend(const bool* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclSend(const char* send_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclRecv(T* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclRecv(recv_buf, data_size, nccl_data_type, peer, nccl_param.nccl_comm_, stream));
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template void
ftNcclRecv(float* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclRecv(half* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
ftNcclRecv(__nv_bfloat16* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
#endif
template void ftNcclRecv(int* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclRecv(bool* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclRecv(char* recv_buf, const int data_size, const int peer, NcclParam nccl_param, cudaStream_t stream);

template<typename T>
void ftNcclBroadCast(T* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclBcast(buff, data_size, nccl_data_type, root, nccl_param.nccl_comm_, stream));
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

template void
ftNcclBroadCast(char* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(bool* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(int* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(float* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
template void
ftNcclBroadCast(half* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
#ifdef ENABLE_BF16
template void
ftNcclBroadCast(__nv_bfloat16* buff, const int data_size, const int root, NcclParam nccl_param, cudaStream_t stream);
#endif

template void ftNcclAllReduceSum(
    const float* send_buf, float* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

template void ftNcclAllReduceSum(
    const half* send_buf, half* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

template void ftNcclAllReduceSum(
    const int32_t* send_buf, int32_t* recv_buf, const int data_size, NcclParam nccl_param, cudaStream_t stream);

#ifdef ENABLE_BF16
template void ftNcclAllReduceSum(const __nv_bfloat16* send_buf,
                                 __nv_bfloat16*       recv_buf,
                                 const int            data_size,
                                 NcclParam            nccl_param,
                                 cudaStream_t         stream);
#endif

template void ftNcclAllGather(const float* send_buf,
                              float*       recv_buf,
                              const int    data_size,
                              const int    rank,
                              NcclParam    nccl_param,
                              cudaStream_t stream);

template void ftNcclAllGather(const half*  send_buf,
                              half*        recv_buf,
                              const int    data_size,
                              const int    rank,
                              NcclParam    nccl_param,
                              cudaStream_t stream);

#ifdef ENABLE_BF16
template void ftNcclAllGather(const __nv_bfloat16* send_buf,
                              __nv_bfloat16*       recv_buf,
                              const int            data_size,
                              const int            rank,
                              NcclParam            nccl_param,
                              cudaStream_t         stream);
#endif

void ftNcclGroupStart()
{
#ifdef BUILD_MULTI_GPU
    NCCLCHECK(ncclGroupStart());
#endif
}

void ftNcclGroupEnd()
{
#ifdef BUILD_MULTI_GPU
    NCCLCHECK(ncclGroupEnd());
#endif
}

void ftNcclBarrier(NcclParam param, cudaStream_t stream)
{
// since there are no barriers in NCCL, act as a barrier
#ifdef BUILD_MULTI_GPU
    void* buffer;
    cudaMalloc(&buffer,sizeof(int));
    NCCLCHECK(ncclAllReduce(buffer, buffer, 1, ncclInt, ncclSum, param.nccl_comm_, stream));
    cudaStreamSynchronize(stream);
#endif
}

void ftNcclStreamSynchronize(NcclParam tensor_para, NcclParam pipeline_para, cudaStream_t stream)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    cudaError_t  cudaErr;
    ncclResult_t tensor_ncclErr = ncclSuccess, tensor_ncclAsyncErr = ncclSuccess, pipeline_ncclErr = ncclSuccess,
                 pipeline_ncclAsyncErr = ncclSuccess;
    ncclComm_t tensor_comm             = tensor_para.nccl_comm_;
    ncclComm_t pipeline_comm           = pipeline_para.nccl_comm_;
    if (tensor_para.world_size_ == 1 && pipeline_para.world_size_ == 1) {
        check_cuda_error(cudaStreamSynchronize(stream));
        return;
    }
    while (1) {
        cudaErr = cudaStreamQuery(stream);
        if (cudaErr == cudaSuccess) {
            FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
            return;
        }

        if (cudaErr != cudaErrorNotReady) {
            std::string error_msg = "CUDA Error : cudaStreamQuery returned " + std::to_string(cudaErr);
            throw std::runtime_error(error_msg);
        }
        if (tensor_para.world_size_ > 1) {
            tensor_ncclErr = ncclCommGetAsyncError(tensor_comm, &tensor_ncclAsyncErr);
        }
        if (pipeline_para.world_size_ > 1) {
            pipeline_ncclErr = ncclCommGetAsyncError(pipeline_comm, &pipeline_ncclAsyncErr);
        }

        if (tensor_ncclErr != ncclSuccess || pipeline_ncclErr != ncclSuccess) {
            std::string error_msg = "NCCL Error : ncclCommGetAsyncError returned " + std::to_string(tensor_ncclErr)
                                    + " (tensor_para) " + std::to_string(pipeline_ncclErr) + " (pipeline_para)";
            throw std::runtime_error(error_msg);
        }

        if (tensor_ncclAsyncErr != ncclSuccess) {
            // An asynchronous error happened. Stop the operation and destroy
            // the communicator
            tensor_ncclErr = ncclCommAbort(tensor_comm);
            if (tensor_ncclErr != ncclSuccess) {
                std::string error_msg = "NCCL Error : ncclCommDestroy returned " + std::to_string(tensor_ncclErr);
                throw std::runtime_error(error_msg);
            }
        }

        if (pipeline_ncclAsyncErr != ncclSuccess) {
            // An asynchronous error happened. Stop the operation and destroy
            // the communicator
            pipeline_ncclErr = ncclCommAbort(pipeline_comm);
            if (pipeline_ncclErr != ncclSuccess) {
                std::string error_msg = "NCCL Error : ncclCommDestroy returned " + std::to_string(pipeline_ncclErr);
                throw std::runtime_error(error_msg);
            }
        }
    }
#endif
}

void ftNcclGetUniqueId(NcclUid& uid)
{
#ifdef BUILD_MULTI_GPU
    NCCLCHECK(ncclGetUniqueId(&uid.nccl_uid_));
#endif
}

void ftNcclCommInitRank(NcclParam& param, const int rank, const int world_size, const NcclUid uid)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    // Initialize a nccl communicator.
    if (param.nccl_comm_ != nullptr) {
        FT_LOG_WARNING("NcclParam is already initialized.");
        return;
    }
    param.rank_       = rank;
    param.world_size_ = world_size;
    param.nccl_uid_   = uid.nccl_uid_;
    NCCLCHECK(ncclCommInitRank(&param.nccl_comm_, param.world_size_, param.nccl_uid_, param.rank_));
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void ftNcclParamDestroy(NcclParam& param)
{
#ifdef BUILD_MULTI_GPU
    if (param.nccl_comm_ != nullptr) {
        ncclCommDestroy(param.nccl_comm_);
    }
#endif
}

void ftNcclInitialize(NcclParam& tensor_para,
                      NcclParam& pipeline_para,
                      NcclParam& cache_stream_para,
                      const int  tensor_para_size,
                      const int  pipeline_para_size,
                      const int  prompt_world_size,
                      const int  token_world_size,
                      const int  world_size,
                      const int  rank,
                      const bool with_mpi)
{

    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    // Initialize nccl communication grid of tensor and pipeline parallel groups.
#ifndef BUILD_MULTI_GPU
    FT_CHECK_WITH_INFO(tensor_para_size == 1,
                       fmtstr("tensor_para_size=%d although BUILD_MULTI_GPU is disabled. "
                              "Please use the cmake flag -DBUILD_MULTI_GPU=ON if you want "
                              "to use tensor/pipeline parallelism.",
                              tensor_para_size));
    FT_CHECK_WITH_INFO(pipeline_para_size == 1,
                       fmtstr("pipeline_para_size=%d although BUILD_MULTI_GPU is disabled. "
                              "Please use the cmake flag -DBUILD_MULTI_GPU=ON if you want "
                              "to use tensor/pipeline parallelism.",
                              pipeline_para_size));
    tensor_para.rank_         = 0;
    tensor_para.world_size_   = tensor_para_size;
    pipeline_para.rank_       = 0;
    pipeline_para.world_size_ = pipeline_para_size;
#else
    // Initialize a nccl communicator.
    if (tensor_para.nccl_comm_ != nullptr && pipeline_para.nccl_comm_ != nullptr) {
        FT_LOG_WARNING("NcclParam is already initialized. Skip NCCL initialization.");
        return;
    }
    FT_CHECK(tensor_para.nccl_comm_ == nullptr);
    FT_CHECK(pipeline_para.nccl_comm_ == nullptr);
    FT_CHECK(tensor_para_size > 0);
    FT_CHECK(pipeline_para_size > 0);

    // Convert WORLD communicator into 2D grid (k * n) communicator.
    //  row = a tensor parallel group, col = a pipeline parallel group.
    ncclComm_t new_comm, grid_comm, tp_comm, pp_comm;
    ncclUniqueId tp_id, pp_id;
    int tp_rank, pp_rank;

    printf("At nccl init, %d, %d, %d, %d\n", rank, world_size, prompt_world_size, token_world_size);


    if (with_mpi) {
        int mpi_initialized;
        MPICHECK(MPI_Initialized(&mpi_initialized));
        FT_CHECK_WITH_INFO(mpi_initialized, "Fail to nccl initialization because MPI is not initialized.");

        // Convert WORLD communicator into 2D grid (k * n) communicator.
        //  row = a tensor parallel group, col = a pipeline parallel group.
        MPI_Comm new_comm, grid_comm, tp_comm, pp_comm;
        int dims[2]    = {pipeline_para_size, tensor_para_size};
        int periods[2] = {0, 0};

        // used for prompt -> token streaming
#if defined(NCCL_SEND) || defined(MPI_SEND)
        MPI_Comm_split(MPI_COMM_WORLD, rank / (pipeline_para_size * tensor_para_size), rank, &new_comm);
        MPI_Cart_create(new_comm, 2, dims, periods, 0, &grid_comm);
#elif defined(SEPERATE_PROMPT)
        printf("RANK %d IN SEPERATE_PROMPT!\n", rank);
        if (rank < prompt_world_size)
            MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &new_comm);
        else
            MPI_Comm_split(MPI_COMM_WORLD, 1, rank, &new_comm);
        MPI_Cart_create(new_comm, 2, dims, periods, 0, &grid_comm);
#else
        MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 0, &grid_comm);
#endif

        // Split 2D communicator into rows and cols.
        int tp_remain_dims[2] = {false, true};
        int pp_remain_dims[2] = {true, false};

        MPI_Cart_sub(grid_comm, tp_remain_dims, &tp_comm);
        printf("RANK %d IN MPI_Cart_sub 1!\n", rank);

        MPI_Cart_sub(grid_comm, pp_remain_dims, &pp_comm);
        printf("RANK %d IN MPI_Cart_sub 2!\n", rank);

        MPI_Comm_rank(tp_comm, &tp_rank);
        MPI_Comm_rank(pp_comm, &pp_rank);

        printf("Ranks: %d, %d, %d\n", rank, tp_rank, pp_rank);

        ncclUniqueId tp_uid;
        ncclUniqueId pp_uid;

        // for NCCL
        // std::this_thread::sleep_for(std::chrono::milliseconds(rank*1000));

        // The root of each group creates a nccl uid.
        if (tp_rank == 0) {
            FT_LOG_DEBUG("---------- rank %d tp rank %d creates nccl uid.\n", rank, tp_rank);
            NCCLCHECK(ncclGetUniqueId(&tp_uid));
        }

        if (pp_rank == 0) {
            FT_LOG_DEBUG("************ rank %d pp rank %d creates nccl uid.\n", rank, pp_rank);
            NCCLCHECK(ncclGetUniqueId(&pp_uid));
        }

        // Broadcast nccl uid to share the same nccl uid across gpus in the same group.

        MPI_Bcast(&tp_uid, sizeof(tp_uid), MPI_BYTE, 0, tp_comm);
        MPI_Bcast(&pp_uid, sizeof(pp_uid), MPI_BYTE, 0, pp_comm);
    }
    else {
        std::string local_ip = "127.0.0.1";
        int port = std::stoi(std::getenv("MASTER_PORT")) + 1;
        const char* ip_addr;
        int pp_group;


        if (rank < prompt_world_size) {
            ip_addr = std::getenv("PROMPT_MASTER_ADDR");
            pp_group = rank % tensor_para_size;
            pp_rank = rank / tensor_para_size;
        }
        else {
            int offset_rank = rank - prompt_world_size;
            int prompt_pipeline_size = prompt_world_size / tensor_para_size;
            pp_group = prompt_pipeline_size + offset_rank % tensor_para_size;
            pp_rank = offset_rank / tensor_para_size;
            ip_addr = std::getenv("TOKEN_MASTER_ADDR");
        }

        std::string ip(ip_addr);

        // TODO: adapt for streaming
        tp_rank = rank % tensor_para_size;

        int tp_group = rank / tensor_para_size;

        initNcclUniqueId(&tp_id, local_ip, port + tp_group);
        initNcclUniqueId(&pp_id, ip, port + pipeline_para_size + pp_group);
    }

    NCCLCHECK(ncclCommInitRank(&tp_comm, tensor_para_size, tp_id, tp_rank));
    NCCLCHECK(ncclCommInitRank(&pp_comm, pipeline_para_size, pp_id, pp_rank));

    tensor_para.world_size_   = tensor_para_size;
    tensor_para.rank_         = tp_rank;
    tensor_para.nccl_comm_    = tp_comm;
    pipeline_para.world_size_ = pipeline_para_size;
    pipeline_para.rank_       = pp_rank;
    pipeline_para.nccl_comm_  = pp_comm;
    printf("NCCL initialized rank=%d world_size=%d tensor_para=%s pipeline_para=%s\n",
           rank,
           world_size,
           tensor_para.toString().c_str(),
           pipeline_para.toString().c_str());
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

void initNcclUniqueId(ncclUniqueId* ncclId, std::string ip_addr, uint16_t port) {

    memset(ncclId->internal, 0, sizeof(ncclId->internal));
    // NET
    ncclId->internal[0] = 2;

    // PORT
    uint16_t* ptr = (uint16_t*)(&(ncclId->internal[2]));
    *ptr = port;

    std::cout << ip_addr << "," << port << std::endl;

    // IP
    std::vector<std::string> ip_tokens;
    boost::split(ip_tokens, ip_addr, boost::is_any_of("."));

    ncclId->internal[4] = std::stoi(ip_tokens[0]);
    ncclId->internal[5] = std::stoi(ip_tokens[1]);
    ncclId->internal[6] = std::stoi(ip_tokens[2]);
    ncclId->internal[7] = std::stoi(ip_tokens[3]);

}

void ftNcclCacheInitialize(NcclParam& cache_para, const int cache_para_size, const int torch_rank, const bool with_mpi)
{

    FT_CHECK(cache_para.nccl_comm_ == nullptr);
    FT_CHECK(cache_para_size > 0);
    int world_size = cache_para_size;
    printf("At cache init, %d, %d, %d\n", torch_rank, world_size, with_mpi);
    ncclUniqueId cs_uid;
    ncclComm_t cs_nccl_comm;


    if (with_mpi) {
        // MPI is initialized by this point
        int mpi_initialized;
        MPICHECK(MPI_Initialized(&mpi_initialized));
        FT_CHECK_WITH_INFO(mpi_initialized, "Fail to nccl initialization because MPI is not initialized.");

        if (torch_rank == 0) {
            NCCLCHECK(ncclGetUniqueId(&cs_uid));
        }

        printf("Rank %d, Broadcast nccl uid to the others in the same parallel groups.\n", torch_rank);
        MPI_Bcast(&cs_uid, sizeof(cs_uid), MPI_BYTE, 0, MPI_COMM_WORLD);

    }
    else {
        const char* ip_addr = std::getenv("MASTER_ADDR");
        int port = std::stoi(std::getenv("MASTER_PORT"));
        std::string ip(ip_addr);

        initNcclUniqueId(&cs_uid, ip, port+1);

        printf("Initialize NCCL communicator for cache streaming.\n");

    }

    printf("BEFORE ncclCommInitRank!!\n");
    NCCLCHECK(ncclCommInitRank(&cs_nccl_comm, world_size, cs_uid, torch_rank));
    printf("AFTER ncclCommInitRank!!\n");

    cache_para.world_size_ = world_size;
    cache_para.rank_       = torch_rank;
    cache_para.nccl_uid_   = cs_uid;
    cache_para.nccl_comm_  = cs_nccl_comm;

    printf("Cache world size is %d, rank is %d. uid is %d\n",
           cache_para.world_size_,
           cache_para.rank_,
           cache_para.nccl_uid_);
}

size_t getLocalBatchSize(const size_t batch_size, const size_t seq_len, const size_t pipeline_para_size)
{
    size_t local_batch_size = batch_size;
    if (pipeline_para_size == 1) {
        return local_batch_size;
    }
    if (local_batch_size % pipeline_para_size == 0) {
        local_batch_size /= pipeline_para_size;
    }

    while (local_batch_size * seq_len > 1024 && local_batch_size % 2 == 0) {
        local_batch_size /= 2;
    }
    return local_batch_size;
}

}  // namespace fastertransformer
