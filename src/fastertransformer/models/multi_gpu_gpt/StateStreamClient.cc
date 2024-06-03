#include "src/fastertransformer/models/multi_gpu_gpt/StateStreamClient.h"

using namespace grpc;
using namespace dejavu;

DejaVuClient::DejaVuClient(std::shared_ptr<Channel> channel)
	: stub_(DejaVuManager::NewStub(channel)) {}

int DejaVuClient::GetSlot() {
	PushRequest req;
	PushResponse resp;
	return GetSlotImpl(req, &resp);
}

void DejaVuClient::MarkComplete(int slot_id) {
	CompleteRequest req;
	CompleteResponse resp;
	req.set_slot_id(slot_id);
	MarkCompleteImpl(req, &resp);
}

int DejaVuClient::GetSlotImpl(PushRequest &req, PushResponse *resp) {
	ClientContext context;
	std::chrono::system_clock::time_point deadline = std::chrono::system_clock::now() + std::chrono::seconds(1);
	context.set_deadline(deadline);

	Status status = stub_->Push(&context, req, resp);
	if (status.ok()) {
		return resp->slot_id();
	}
	return -1;
}

void DejaVuClient::MarkCompleteImpl(CompleteRequest &req, CompleteResponse *resp) {
	ClientContext context;
	Status status = stub_->Complete(&context, req, resp);
	assert(status.ok());
}
