# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from protos import api_server_pb2 as protos_dot_api__server__pb2


class ApiServerStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.Serve = channel.unary_unary(
                '/dejavu_api.ApiServer/Serve',
                request_serializer=protos_dot_api__server__pb2.InferenceRequest.SerializeToString,
                response_deserializer=protos_dot_api__server__pb2.InferenceResponse.FromString,
                )
        self.ServeOpen = channel.unary_unary(
                '/dejavu_api.ApiServer/ServeOpen',
                request_serializer=protos_dot_api__server__pb2.InferenceRequestUbatch.SerializeToString,
                response_deserializer=protos_dot_api__server__pb2.InferenceResponseOpen.FromString,
                )


class ApiServerServicer(object):
    """Missing associated documentation comment in .proto file."""

    def Serve(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def ServeOpen(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ApiServerServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'Serve': grpc.unary_unary_rpc_method_handler(
                    servicer.Serve,
                    request_deserializer=protos_dot_api__server__pb2.InferenceRequest.FromString,
                    response_serializer=protos_dot_api__server__pb2.InferenceResponse.SerializeToString,
            ),
            'ServeOpen': grpc.unary_unary_rpc_method_handler(
                    servicer.ServeOpen,
                    request_deserializer=protos_dot_api__server__pb2.InferenceRequestUbatch.FromString,
                    response_serializer=protos_dot_api__server__pb2.InferenceResponseOpen.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'dejavu_api.ApiServer', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ApiServer(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def Serve(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/dejavu_api.ApiServer/Serve',
            protos_dot_api__server__pb2.InferenceRequest.SerializeToString,
            protos_dot_api__server__pb2.InferenceResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def ServeOpen(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/dejavu_api.ApiServer/ServeOpen',
            protos_dot_api__server__pb2.InferenceRequestUbatch.SerializeToString,
            protos_dot_api__server__pb2.InferenceResponseOpen.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
