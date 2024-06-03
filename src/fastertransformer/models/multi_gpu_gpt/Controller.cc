#include "src/fastertransformer/models/multi_gpu_gpt/Controller.h"

using namespace grpc;
using namespace std::chrono;

Status ControllerImpl::Reset(ServerContext* context, const ResetRequest* request, ResetResponse* response)
{

    printf("RESET, %d\n", local_reset_);
    for (int i = 0; i < num_peers_; i++) {
        health_status_[i] = high_resolution_clock::now();
        alive_[i]         = true;
        health_set_[i]    = false;
    }
    dead_found_   = false;
    min_step_set_ = false;
    min_steps_.empty();

    std::atomic_thread_fence(std::memory_order_seq_cst);
    if (local_reset_ > 0) {
        printf("A FAILURE HAPPENED, RESET!\n");
        *reset_ = 1;
    }
    local_reset_++;
    printf("RESET 2, %d\n", local_reset_);

    return Status::OK;
}

Status
ControllerImpl::SendHeartbeat(ServerContext* context, const HeartBeatRequest* request, HeartBeatResponse* response)
{
    health_status_[request->rank()] = high_resolution_clock::now();
    printf("RECEIVED A HEARTBEAT FROM RANK %d\n", request->rank());

    auto                         time_now    = high_resolution_clock::now();
    duration<double, std::milli> duration_ms = time_now - last_check_;
    if (dead_found_ || duration_ms.count() >= 3 * ms_heartbeat_) {
        // check only for tokens
        for (int i = num_prompt_peers_; i < num_peers_; i++) {
            if (health_set_[i]) {  // ignore first heartbeat
                std::chrono::duration<double, std::milli> ms_double = time_now - health_status_[i];
                if (ms_double.count() > 3 * ms_heartbeat_) {
                    printf("RANK %d must be dead!\n", i);
                    alive_[i]   = false;
                    dead_found_ = true;
                }
            }
        }
        last_check_ = high_resolution_clock::now();
    }
    health_set_[request->rank()] = true;
    printf("DEAD FOUND IS %d\n", dead_found_);
    // TODO: what to do here?

    if (!dead_found_) {
        response->set_reset(false);
    }
    else {
        response->set_reset(true);
    }

    return Status::OK;
}

Status ControllerImpl::IsReady(ServerContext* context, const IsReadyRequest* request, IsReadyResponse* response)
{

    is_ready_[request->rank()] = true;
    bool all_ready             = true;

    for (auto it : is_ready_)
        all_ready &= it;

    if (all_ready) {
        ctrl_mtx_->lock();
        *workers_ready_ = true;
        ctrl_mtx_->unlock();
    }

    return Status::OK;
}

Status ControllerImpl::GetInfo(ServerContext* context, const StartUpInfoRequest* request, StartUpInfoResponse* response)
{

    printf("[CONFIG] Received from rank %d\n", request->rank());

    if (!with_ft_) {
        *response->mutable_ubatch_global_ids() = {active_ubatches_.begin(),
                                                  active_ubatches_.end()};  // TODO: adapt for prompt!
        std::vector<int> v(num_token_peers_, 0);
        *response->mutable_ubatch_steps() = {v.begin(), v.end()};

        response->set_has_failed(false);
        response->set_stream_cache_next(false);
        response->set_stream_cache_prev(false);

        return Status::OK;
    }

    if (!min_step_set_) {
        // after restart
        auto time_now = high_resolution_clock::now();
        if (dead_found_) {
            for (int i = 0; i < num_token_peers_; i++) {
                min_steps_[i] = INT_MAX;
                for (int j = num_prompt_peers_; j < num_peers_; j++) {
                    min_steps_[i] = std::min(min_steps_[i], cache_replica_status_[j][active_ubatches_[i]]);
                }
            }
            min_step_set_ = true;
        }
    }

    int min_value = *std::min_element(min_steps_.begin(), min_steps_.end());
    ;
    for (int i = 0; i < num_token_peers_; i++)
        min_steps_[i] = min_value;

    printf("[CONFIG] After finding the latest step\n");
    for (int i = 0; i < num_token_peers_; i++) {
        printf("%d, %d\n", i, min_steps_[i]);
    }

    int rank    = request->rank();
    bool stream_cache_next = false;
    bool stream_cache_prev = false;

    if (rank >= num_prompt_peers_) {

        int token_rank = rank - num_prompt_peers_;
        int pp_rank = token_rank / tensor_para_size_;
        int tp_rank = token_rank % tensor_para_size_;

        printf("Check for rank %d\n", rank);


        int next_rank = num_prompt_peers_ + ((pp_rank + 1) * tensor_para_size_) % num_token_peers_ + tp_rank;
        if (alive_[rank] && !alive_[next_rank])
            stream_cache_next = true;

        int prev_rank = 0;

        if (alive_[rank]) {
            prev_rank = num_prompt_peers_ + (pp_rank - 1) * tensor_para_size_ + tp_rank;
            if (pp_rank == 0)
                prev_rank = num_prompt_peers_ + (token_pipeline_size_ - 1) * tensor_para_size_ + tp_rank;

            if (!alive_[prev_rank])
                stream_cache_prev = true;
        }

    }

    printf("Config to start with: \n");
    for (int i = 0; i < num_token_peers_; i++) {
        printf("Ubatch %d, step %d\n", active_ubatches_[i], min_steps_[i]);
    }

    *response->mutable_ubatch_global_ids() = {active_ubatches_.begin(),
                                              active_ubatches_.end()};  // TODO: adapt for prompt!
    *response->mutable_ubatch_steps()      = {min_steps_.begin(), min_steps_.end()};
    *response->mutable_prompts_seen()      = {prompt_seen_global_ids_.begin(), prompt_seen_global_ids_.end()};

    response->set_has_failed(!alive_[rank]);
    response->set_stream_cache_next(stream_cache_next);
    response->set_stream_cache_prev(stream_cache_prev);

    return Status::OK;
}

Status ControllerImpl::SendCacheAck(ServerContext* context, const CacheRequest* request, CacheResponse* response)
{
    cache_replica_status_[request->rank()][request->ubatch_id()] = request->step();
    // printf("[CACHE ACK] Received from rank %d, ubatch %d, step %d\n", request->rank(), request->ubatch_id(),
    // request->step());
    return Status::OK;
}

Status ControllerImpl::MarkRestart(ServerContext* context, const IsRestartRequest* request, IsRestartResponse* response)
{
    printf("Peer %d found dead and restarted\n", request->rank());
    alive_[request->rank()] = false;
    dead_found_             = true;
    return Status::OK;
}

Status ControllerImpl::SendToken(ServerContext* context, const TokenRequest* request, TokenResponse* response)
{
    std::vector<int> tokens(request->tokens().begin(), request->tokens().end());
    if (ubatch_tokens_.find(request->ubatch_id()) == ubatch_tokens_.end())
        ubatch_tokens_[request->ubatch_id()] = {};
    ubatch_tokens_[request->ubatch_id()].push_back(tokens);
    std::string tokens_str = "";
    for (auto token : tokens) {
        tokens_str += std::to_string(token);
        tokens_str += ",";
    }

    // token_times_.push_back(std::chrono::duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count());

    active_ubatches_[request->pp_id()] = request->ubatch_id();
    // for (auto it: active_ubatches_)
    //     std::cout << it << ",";
    // std::cout << std::endl;

    // printf("[TOKENS] Received for ubatch %d, tokens: %s\n", request->ubatch_id(), tokens_str.c_str());

    return Status::OK;
}

Status ControllerImpl::MarkUbatchFinished(ServerContext*               context,
                                          const UbatchFinishedRequest* request,
                                          UbatchFinishedResponse*      response)
{
    // while ((change_->load())) ;

    printf("RECEIVED A MarkUbatchFinished!\n");

    std::vector<int> finished_ids(request->ubatch_id().begin(), request->ubatch_id().end());
    std::vector<int> finished_pp_ids(request->pp_id().begin(), request->pp_id().end());
    int              rank = request->rank();

    // clear from structures
    int num_requests = finished_ids.size();
    for (int j = 0; j < num_requests; j++) {
        for (int i = num_prompt_peers_; i < num_peers_; i++) {
            cache_replica_status_[i].erase(finished_ids[j]);
        }
        if (rank == num_peers_ - 1) {
            printf("RANK %d, UBATCH %d, %d DONE\n", request->rank(), finished_ids[j], finished_pp_ids[j]);
        }
        else {
            prompt_seen_global_ids_.insert(finished_ids[j]);
            printf("RANK %d, UBATCH %d, PROMPT DONE\n", request->rank(), finished_ids[j]);
        }
    }
    if (rank == num_peers_ - 1) {
        ctrl_mtx_->lock();
        for (auto it : finished_ids) {
            finished_reqs_->push_back(it);
            end_times_[(int64_t)it] =
                std::chrono::duration_cast<seconds>(system_clock::now().time_since_epoch()).count();
        }
        ctrl_mtx_->unlock();

        std::atomic_thread_fence(std::memory_order_seq_cst);
        *change_ = 1;
    }

    return Status::OK;
}

void RunController(ControllerImpl&    service,
                   std::string        ip_str,
                   int                port,
                   int                num_peers,
                   int                num_prompt_peers,
                   int                num_token_peers,
                   int                tensor_para_size,
                   bool               with_ft,
                   std::atomic<bool>* finished,
                   std::atomic<bool>* change,
                   std::atomic<bool>* reset,
                   std::vector<int>*  finished_reqs,
                   std::atomic<bool>* workers_ready,
                   std::mutex*        ctrl_mtx)
{
    // TODO: change this
    std::string port_str      = std::to_string(port);
    std::string total_address = ip_str + ":" + port_str;
    std::string server_address(total_address.c_str());

    service.num_peers_        = num_peers;
    service.num_token_peers_  = num_token_peers;
    service.num_prompt_peers_ = num_prompt_peers;
    service.tensor_para_size_ = tensor_para_size;

    service.prompt_pipeline_size_ = num_prompt_peers / tensor_para_size;
    service.token_pipeline_size_ = num_token_peers / tensor_para_size;

    assert(service.num_peers_ == service.num_token_peers_ + service.num_prompt_peers_);

    service.with_ft_       = with_ft;
    service.finished_      = finished;
    service.change_        = change;
    service.reset_         = reset;
    service.local_reset_   = 0;
    service.finished_reqs_ = finished_reqs;
    service.ctrl_mtx_      = ctrl_mtx;
    service.workers_ready_ = workers_ready;

    printf("Config: %d, %d, %d, %d\n",
           service.num_peers_,
           service.num_prompt_peers_,
           service.num_token_peers_,
           service.with_ft_);

    for (int i = 0; i < num_peers; i++) {
        service.health_status_.push_back(high_resolution_clock::now());
        service.cache_replica_status_.push_back({});
        service.health_set_.push_back(false);
        service.alive_.push_back(true);
        service.is_ready_.push_back(false);
    }

    for (int i = 0; i < num_token_peers; i++) {
        service.active_ubatches_.push_back(0);
        service.min_steps_.push_back(0);
    }

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    // TODO: check number of threads here
    service.server = std::unique_ptr<Server>(builder.BuildAndStart());
    std::cout << "Controller listening on " << server_address << std::endl;
    service.server->Wait();
}

void Shutdown(ControllerImpl& service)
{
    // for (auto it: service.ubatch_tokens_) {
    //     printf("Ubatch %d: ", it.first);
    //     for (auto t: it.second) {
    //         std::cout << t[0] << ",";
    //     }
    //     std::cout << std::endl;
    // }
    service.health_status_.empty();
    service.cache_replica_status_.empty();
    service.health_set_.empty();
    service.ubatch_tokens_.empty();
    service.server->Shutdown();
}
