#include "src/fastertransformer/models/multi_gpu_gpt/StateStreamServer.h"


using namespace grpc;
using namespace dejavu;

Status DejaVuManagerImpl::Push(ServerContext* context, const PushRequest* request, PushResponse* response)
{

    avail_mtx_.lock();
    if (!avail_queue_.empty()) {
        response->set_slot_id(avail_queue_.front());
        avail_queue_.pop();
    }
    else {
        response->set_slot_id(-1);
    }
    avail_mtx_.unlock();
    return Status::OK;
}

Status DejaVuManagerImpl::Complete(ServerContext* context, const CompleteRequest* request, CompleteResponse* response)
{

    printf("[SERVER] Add %d to written queue\n", request->slot_id());
    written_mtx_.lock();
    written_queue_.push(request->slot_id());
    written_mtx_.unlock();
    return Status::OK;
}

void RunServer(DejaVuManagerImpl& service, int port, int num_microbatches, int prompt_buffer_size)
{
    // TODO: change this
    std::string port_str = std::to_string(port);
    char ip_address[INET_ADDRSTRLEN];
    int          fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_IP);
    struct ifreq ifr {};
    strcpy(ifr.ifr_name, "eth0");
    ioctl(fd, SIOCGIFADDR, &ifr);
    close(fd);

    strcpy(ip_address, inet_ntoa(((sockaddr_in*)&ifr.ifr_addr)->sin_addr));
    std::string ip_str(ip_address);

    std::string total_address = ip_str + ":" + port_str;
    std::cout << "Start Server at " << total_address << std::endl;

    std::string server_address(total_address.c_str());

    service.prompt_buffer_size_ = prompt_buffer_size;

    // init queue
    service.avail_mtx_.lock();
    for (int i = 0; i < service.prompt_buffer_size_; i++)
        service.avail_queue_.push(i);
    service.avail_mtx_.unlock();
    printf("QUEUE SIZE %d\n", service.avail_queue_.size());

    ServerBuilder builder;
    builder.AddListeningPort(server_address, grpc::InsecureServerCredentials());
    builder.RegisterService(&service);

    // TODO: check number of threads here
    service.dv_server = std::unique_ptr<Server>(builder.BuildAndStart());
    //std::cout << "Server listening on " << server_address << std::endl;
    service.dv_server->Wait();
}

void Shutdown(DejaVuManagerImpl& service) {
    service.dv_server->Shutdown();
}
