import grpc
import sys
import os
import signal
from concurrent import futures

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
from examples.pytorch.gpt.api.protos.api_server_pb2_grpc import ApiServerServicer, add_ApiServerServicer_to_server
from examples.pytorch.gpt.api.protos.api_server_pb2 import InferenceResponse

API_SERVER_PORT = 51000

class ApiServer(ApiServerServicer):
    def __init__(self, lock, worker_ready, input_dict, rank, open_loop=False, input_list=None):
        self.lock_ = lock
        self.input_dict_ = input_dict
        self.input_list_ = input_list
        self.worker_ready_ = worker_ready
        self.rank_ = rank
        self.open_loop_ = open_loop

    def Serve(self, request, context):
        print(f"Rank {self.rank_}, Request received!")
        if not self.open_loop_:
            while True:
                if (self.worker_ready_.value == 1):
                    break

        with self.lock_:
            self.input_dict_['input_ids'] = list(request.input_ids)
            self.input_dict_['input_lengths'] = list(request.input_lengths)
            self.input_dict_['output_lengths'] = list(request.output_lengths)
            self.input_dict_['ubatch_ids'] = list(request.ubatch_ids)

        return InferenceResponse()

    def ServeOpen(self, request, context):
        #print(f"Rank {self.rank_}, Request received, list is {self.input_list_}")
        input_dict = {}
        input_dict['ubatch_id'] = int(request.ubatch_id)
        input_dict['input_lengths'] = list(request.input_lengths)
        input_dict['output_lengths'] = list(request.output_lengths)
        input_dict['input_ids'] = list(request.input_ids)

        #print(input_dict)

        with self.lock_:
            self.input_list_.append(input_dict)

        return InferenceResponse()


def serve(lock, worker_ready, input_dict, rank, open_loop=False, input_list=None):
    print(f"------------- At Serve, Port {API_SERVER_PORT + rank}, {input_list}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_ApiServerServicer_to_server(
        ApiServer(lock, worker_ready, input_dict, rank, open_loop, input_list), server)
    server.add_insecure_port(f'[::]:{API_SERVER_PORT + rank}')

    def terminate(signum,_):
        done = server.stop(5)
        done.wait()
        print(f"Received {signum}, stop complete!")

    server.start()
    signal.signal(signal.SIGTERM, terminate)
    server.wait_for_termination()
    print("------ exit ------")