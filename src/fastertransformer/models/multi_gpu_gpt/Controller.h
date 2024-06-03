#include <mutex>
#include <queue>
#include <set>
#include <unordered_map>
#include <vector>

#include <grpc/grpc.h>
#include <grpc/support/log.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>
#include <grpcpp/grpcpp.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/security/credentials.h>

#include "src/fastertransformer/models/multi_gpu_gpt/ft_state.grpc.pb.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ft_state.pb.h"

using namespace grpc;
using namespace dejavu_ft;

class ControllerImpl: public Controller::Service {

public:
    std::unique_ptr<Server> server;

    double                                         ms_heartbeat_ = 1000.0;
    bool                                           dead_found_   = false;
    int                                            num_peers_;
    int                                            ubatch_step_id_;
    std::chrono::high_resolution_clock::time_point last_check_;

    std::vector<std::chrono::high_resolution_clock::time_point> health_status_;
    std::vector<bool>                                           health_set_;
    std::vector<std::unordered_map<int, int>>                   cache_replica_status_;
    std::unordered_map<int, std::vector<std::vector<int>>>      ubatch_tokens_;
    std::vector<bool>                                           alive_;
    std::vector<int>                                            min_steps_;
    std::vector<int>                                            active_ubatches_;
    std::set<int>                                               prompt_seen_global_ids_;

    bool               min_step_set_ = true;
    std::atomic<bool>* finished_;
    std::atomic<bool>* change_;
    std::atomic<bool>* reset_;
    std::atomic<bool>* workers_ready_;

    std::mutex*                          ctrl_mtx_;
    std::vector<int>*                    finished_reqs_;
    std::unordered_map<int64_t, int64_t> end_times_;
    std::vector<int64_t>                 token_times_;
    std::vector<bool>                    is_ready_;

    int  num_token_peers_;
    int  num_prompt_peers_;
    int  tensor_para_size_;
    int  prompt_pipeline_size_;
    int  token_pipeline_size_;
    int  local_reset_;
    bool with_ft_;

    Status SendHeartbeat(ServerContext* context, const HeartBeatRequest* request, HeartBeatResponse* response);
    Status SendCacheAck(ServerContext* context, const CacheRequest* request, CacheResponse* response);
    Status SendToken(ServerContext* context, const TokenRequest* request, TokenResponse* response);
    Status GetInfo(ServerContext* context, const StartUpInfoRequest* request, StartUpInfoResponse* response);
    Status Reset(ServerContext* context, const ResetRequest* request, ResetResponse* response);
    Status
    MarkUbatchFinished(ServerContext* context, const UbatchFinishedRequest* request, UbatchFinishedResponse* response);
    Status IsReady(ServerContext* context, const IsReadyRequest* request, IsReadyResponse* response);
    Status MarkRestart(ServerContext* context, const IsRestartRequest* request, IsRestartResponse* response);
};

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
                   std::mutex*        ctrl_mtx);
void Shutdown(ControllerImpl& service);
