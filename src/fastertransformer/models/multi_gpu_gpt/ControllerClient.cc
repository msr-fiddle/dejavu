#include "src/fastertransformer/models/multi_gpu_gpt/ControllerClient.h"

using namespace grpc;
using namespace dejavu_ft;

ControllerClient::ControllerClient(std::shared_ptr<Channel> channel)
	: stub_(Controller::NewStub(channel)) {}

bool ControllerClient::SendHb(int rank) {
	HeartBeatRequest req;
	HeartBeatResponse resp;
	req.set_rank(rank);
	printf("[RANK %d] SEND HB!\n", rank);
	SendHbImpl(req, &resp);
	return resp.reset();
}

void ControllerClient::SendNextToken(int ubatch_id, int pp_id, std::vector<int> &tokens) {
	TokenRequest req;
	TokenResponse resp;
	req.set_ubatch_id(ubatch_id);
	req.set_pp_id(pp_id);
	int size = tokens.size();

	// TODO: check this!
	*req.mutable_tokens() = {tokens.begin(), tokens.end()};
	SendNextTokenImpl(req, &resp);
}

void ControllerClient::SendCacheReceived(int rank, int ubatch_id, int step) {
	CacheRequest req;
	CacheResponse resp;
	req.set_rank(rank);
	req.set_ubatch_id(ubatch_id);
	req.set_step(step);
	SendCacheReceivedImpl(req, &resp);
}

StartUpInfoResponse ControllerClient::GetConfigInfo(int rank) {
	StartUpInfoRequest req;
	StartUpInfoResponse resp;
	req.set_rank(rank);
	GetConfigInfoImpl(req, &resp);
	return resp;
}

void ControllerClient::ControllerReset() {
	ResetRequest req;
	ResetResponse resp;
	ControllerResetImpl(req, &resp);
}


void ControllerClient::MarkUbatchFinished(int rank, std::vector<int> ubatch_id, std::vector<int> pp_id) {
	UbatchFinishedRequest req;
	UbatchFinishedResponse resp;
	req.set_rank(rank);
	*req.mutable_ubatch_id() = {ubatch_id.begin(), ubatch_id.end()};
	*req.mutable_pp_id() = {pp_id.begin(), pp_id.end()};
	MarkUbatchFinishedImpl(req, &resp);
}

void ControllerClient::SendReady(int rank) {
	IsReadyRequest req;
	req.set_rank(rank);
	IsReadyResponse resp;
	SendReadyImpl(req, &resp);
}

void ControllerClient::IsRestart(int rank) {
	IsRestartRequest req;
	req.set_rank(rank);
	IsRestartResponse resp;
	IsRestartImpl(req, &resp);
}

void ControllerClient::SendHbImpl(HeartBeatRequest &req, HeartBeatResponse *resp) {
	ClientContext context;
	Status status = stub_->SendHeartbeat(&context, req, resp);
}

void ControllerClient::SendNextTokenImpl(TokenRequest &req, TokenResponse *resp) {
	ClientContext context;
	Status status = stub_->SendToken(&context, req, resp);
}

void ControllerClient::SendCacheReceivedImpl(CacheRequest &req, CacheResponse *resp) {
	ClientContext context;
	Status status = stub_->SendCacheAck(&context, req, resp);
}

void ControllerClient::GetConfigInfoImpl(StartUpInfoRequest &req, StartUpInfoResponse *resp) {
	ClientContext context;
	Status status = stub_->GetInfo(&context, req, resp);
}

void ControllerClient::ControllerResetImpl(ResetRequest &req, ResetResponse *resp) {
	ClientContext context;
	Status status = stub_->Reset(&context, req, resp);
}

void ControllerClient::MarkUbatchFinishedImpl(UbatchFinishedRequest &req, UbatchFinishedResponse *resp) {
	ClientContext context;
	Status status = stub_->MarkUbatchFinished(&context, req, resp);
}

void ControllerClient::SendReadyImpl(IsReadyRequest &req, IsReadyResponse *resp) {
	ClientContext context;
	Status status = stub_->IsReady(&context, req, resp);
}

void ControllerClient::IsRestartImpl(IsRestartRequest &req, IsRestartResponse *resp) {
	ClientContext context;
	Status status = stub_->MarkRestart(&context, req, resp);
}