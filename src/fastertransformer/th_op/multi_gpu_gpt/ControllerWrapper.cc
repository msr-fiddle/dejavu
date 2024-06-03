#include "src/fastertransformer/th_op/multi_gpu_gpt/ControllerWrapper.h"

namespace th = torch;
namespace ft = fastertransformer;

namespace torch_ext {

ControllerWrapper::ControllerWrapper(std::string ip, int port, int num_peers, int num_prompt_peers, int num_token_peers, int tensor_para_size)
{
    ip_               = ip;
    port_             = port;
    num_peers_        = num_peers;
    num_prompt_peers_ = num_prompt_peers;
    num_token_peers_  = num_token_peers;
    tensor_para_size_ = tensor_para_size;

    change_        = 0;
    reset_         = 0;
    workers_ready_ = 0;
}

void ControllerWrapper::start_server(const bool with_ft)
{
    printf("START server!\n");
    controller_thread_ = std::thread(RunController,
                                     std::ref(controller_impl_),
                                     ip_,
                                     port_,
                                     num_peers_,
                                     num_prompt_peers_,
                                     num_token_peers_,
                                     tensor_para_size_,
                                     with_ft,
                                     finished_,
                                     &change_,
                                     &reset_,
                                     &finished_reqs_,
                                     &workers_ready_,
                                     &ctrl_mtx_);
}

void ControllerWrapper::shutdown_server()
{
    printf("STOP server!\n");
    Shutdown(std::ref(controller_impl_));
    controller_thread_.join();
}

bool ControllerWrapper::wait_till_done()
{
    while (1) {
        if (reset_.load()) {
            reset_ = 0;
            return false;
        }
        if (change_.load()) {
            change_ = 0;
            return true;
        }
    }
}

void ControllerWrapper::wait_till_ready()
{
    while (1) {
        if (workers_ready_.load())
            return;
    }
}

void ControllerWrapper::unset_finished(const int64_t i)
{
    finished_[i] = 0;
    __sync_synchronize();
}

std::vector<int64_t> ControllerWrapper::get_new_finished_reqs()
{

    std::vector<int64_t> new_finished_reqs;
    ctrl_mtx_.lock();
    for (auto it : finished_reqs_) {
        new_finished_reqs.push_back((int64_t)it);
    }
    finished_reqs_.clear();
    ctrl_mtx_.unlock();

    return new_finished_reqs;
}

std::unordered_map<int64_t, int64_t> ControllerWrapper::get_finish_times()
{
    return controller_impl_.end_times_;
}

std::vector<int64_t> ControllerWrapper::get_token_gen_times()
{
    return controller_impl_.token_times_;
}

}  // namespace torch_ext

static auto fasterTransformerControllerWrapperTHS =
    torch::jit::class_<torch_ext::ControllerWrapper>("FasterTransformer", "ControllerWrapper")
        .def(torch::jit::init<std::string, int64_t, int64_t, int64_t, int64_t, int64_t>())
        .def("start_server", &torch_ext::ControllerWrapper::start_server)
        .def("shutdown_server", &torch_ext::ControllerWrapper::shutdown_server)
        .def("unset_finished", &torch_ext::ControllerWrapper::unset_finished)
        .def("wait_till_done", &torch_ext::ControllerWrapper::wait_till_done)
        .def("wait_till_ready", &torch_ext::ControllerWrapper::wait_till_ready)
        .def("get_new_finished_reqs", &torch_ext::ControllerWrapper::get_new_finished_reqs)
        .def("get_finish_times", &torch_ext::ControllerWrapper::get_finish_times)
        .def("get_token_gen_times", &torch_ext::ControllerWrapper::get_token_gen_times);