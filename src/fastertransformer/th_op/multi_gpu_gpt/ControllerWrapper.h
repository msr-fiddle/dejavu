#include <thread>
#include <unordered_map>

#include "src/fastertransformer/models/multi_gpu_gpt/Controller.h"
#include "src/fastertransformer/th_op/th_utils.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

class ControllerWrapper: public th::jit::CustomClassHolder {

    std::string        ip_;
    int                port_;
    int                num_peers_;
    int                num_prompt_peers_;
    int                num_token_peers_;
    int                tensor_para_size_;
    int                pipeline_size_;
    ControllerImpl     controller_impl_;
    std::thread        controller_thread_;
    std::atomic<bool>* finished_;
    std::atomic<bool>  change_;
    std::atomic<bool>  reset_;
    std::atomic<bool>  workers_ready_;

    std::mutex       ctrl_mtx_;
    std::vector<int> finished_reqs_;

public:
    ControllerWrapper(
        std::string ip, int port, int num_peers, int num_prompt_peers, int num_token_peers, int tensor_para_size);
    ~ControllerWrapper() {}

    void                                 start_server(const bool with_ft);
    void                                 shutdown_server();
    void                                 unset_finished(const int64_t i);
    bool                                 wait_till_done();
    std::vector<int64_t>                 get_new_finished_reqs();
    std::unordered_map<int64_t, int64_t> get_finish_times();
    void                                 wait_till_ready();
    std::vector<int64_t>                 get_token_gen_times();
};
}  // namespace torch_ext