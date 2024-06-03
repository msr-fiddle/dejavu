import os
import sys
import grpc
import time

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(dir_path, "../../.."))
from examples.pytorch.gpt.api.protos.api_server_pb2_grpc import ApiServerStub
from examples.pytorch.gpt.api.protos.api_server_pb2 import InferenceRequest, InferenceRequestUbatch

class ApiClient:
    def __init__(self, ip, worker_port):
        # this would work only with 1 worker
        self.grpc_target = f'{ip}:{worker_port}'
        print(f"Create channel at {self.grpc_target}")
        # self.channel_ = grpc.insecure_channel(self.grpc_target)
        # self.stub_ = ApiServerStub(self.channel_)

    def RunServe(self, input_ids, input_lengths, output_lengths, ubatch_ids):
        input_lengths_1d =  [item for sublist in input_lengths for item in sublist]
        input_ids_1d = []
        for ubatch_list in input_ids:
            input_ids_1d += [item for sublist in ubatch_list for item in sublist]
        try:
            with grpc.insecure_channel(self.grpc_target) as channel:
                self.stub_ = ApiServerStub(channel)
                request = InferenceRequest(
                        input_ids = input_ids_1d,
                        input_lengths = input_lengths_1d,
                        output_lengths = output_lengths,
                        ubatch_ids = ubatch_ids
                )
                self.stub_.Serve(request)
                return True
        except grpc.RpcError as e:
            status_code = e.code()
            details = e.details()

            if status_code == grpc.StatusCode.UNAVAILABLE:
                # Server is unavailable
                print("Error: Server is unavailable.")
            elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                # Deadline exceeded
                print("Error: Deadline exceeded.")
            else:
                # Handle other errors
                print(f"Error: {status_code} - {details}")
            return False

    def RunServeUbatch(self, input_ids, input_lengths, output_lengths, ubatch_id):
        input_ids_1d =  [item for sublist in input_ids for item in sublist]
        try:
            with grpc.insecure_channel(self.grpc_target) as channel:
                self.stub_ = ApiServerStub(channel)
                request = InferenceRequestUbatch(
                        input_ids = input_ids_1d,
                        input_lengths = input_lengths,
                        output_lengths = output_lengths,
                        ubatch_id = ubatch_id
                )
                self.stub_.ServeOpen(request)
                return True
        except grpc.RpcError as e:
            status_code = e.code()
            details = e.details()

            if status_code == grpc.StatusCode.UNAVAILABLE:
                # Server is unavailable
                print("Error: Server is unavailable.")
            elif status_code == grpc.StatusCode.DEADLINE_EXCEEDED:
                # Deadline exceeded
                print("Error: Deadline exceeded.")
            else:
                # Handle other errors
                print(f"Error: {status_code} - {details}")
            return False
