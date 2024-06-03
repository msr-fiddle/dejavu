#include <grpc/grpc.h>
#include <grpcpp/channel.h>
#include <grpcpp/client_context.h>
#include <grpcpp/create_channel.h>
#include <grpcpp/security/credentials.h>
#include <grpcpp/health_check_service_interface.h>
#include <grpcpp/ext/proto_server_reflection_plugin.h>

#include "src/fastertransformer/models/multi_gpu_gpt/ft_state.pb.h"
#include "src/fastertransformer/models/multi_gpu_gpt/ft_state.grpc.pb.h"

using namespace grpc;
using namespace dejavu_ft;

class ControllerClient {

	private:
		std::unique_ptr<Controller::Stub> stub_;
	public:
		ControllerClient(std::shared_ptr<Channel> channel);

		bool SendHb(int rank);
		void SendNextToken(int ubatch_id, int pp_id, std::vector<int> &tokens);
        void SendCacheReceived(int rank, int ubatch_id, int step);
		StartUpInfoResponse GetConfigInfo(int rank);
		void ControllerReset();
		void MarkUbatchFinished(int rank, std::vector<int> ubatch_id, std::vector<int> pp_id);
		void SendReady(int rank);
		void IsRestart(int rank);

		void SendHbImpl(HeartBeatRequest &req, HeartBeatResponse *resp);
		void SendNextTokenImpl(TokenRequest &req, TokenResponse *resp);
        void SendCacheReceivedImpl(CacheRequest &req, CacheResponse *resp);
		void GetConfigInfoImpl(StartUpInfoRequest &req, StartUpInfoResponse *resp);
		void ControllerResetImpl(ResetRequest &req, ResetResponse *resp);
		void MarkUbatchFinishedImpl(UbatchFinishedRequest &req, UbatchFinishedResponse *resp);
		void SendReadyImpl(IsReadyRequest &req, IsReadyResponse *resp);
		void IsRestartImpl(IsRestartRequest &req, IsRestartResponse *resp);

};
