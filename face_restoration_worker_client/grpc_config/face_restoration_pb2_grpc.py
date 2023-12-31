# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc
import os
import sys

sys.path.append(os.path.dirname(__file__))

import face_restoration_pb2 as face__restoration__pb2


class FaceRestorationServiceStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.ApplyFaceRestoration = channel.unary_unary(
            "/face_restoration.FaceRestorationService/ApplyFaceRestoration",
            request_serializer=face__restoration__pb2.FaceRestorationRequest.SerializeToString,
            response_deserializer=face__restoration__pb2.FaceRestorationReply.FromString,
        )


class FaceRestorationServiceServicer(object):
    """Missing associated documentation comment in .proto file."""

    def ApplyFaceRestoration(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details("Method not implemented!")
        raise NotImplementedError("Method not implemented!")


def add_FaceRestorationServiceServicer_to_server(servicer, server):
    rpc_method_handlers = {
        "ApplyFaceRestoration": grpc.unary_unary_rpc_method_handler(
            servicer.ApplyFaceRestoration,
            request_deserializer=face__restoration__pb2.FaceRestorationRequest.FromString,
            response_serializer=face__restoration__pb2.FaceRestorationReply.SerializeToString,
        ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
        "face_restoration.FaceRestorationService", rpc_method_handlers
    )
    server.add_generic_rpc_handlers((generic_handler,))


# This class is part of an EXPERIMENTAL API.
class FaceRestorationService(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def ApplyFaceRestoration(
        request,
        target,
        options=(),
        channel_credentials=None,
        call_credentials=None,
        insecure=False,
        compression=None,
        wait_for_ready=None,
        timeout=None,
        metadata=None,
    ):
        return grpc.experimental.unary_unary(
            request,
            target,
            "/face_restoration.FaceRestorationService/ApplyFaceRestoration",
            face__restoration__pb2.FaceRestorationRequest.SerializeToString,
            face__restoration__pb2.FaceRestorationReply.FromString,
            options,
            channel_credentials,
            insecure,
            call_credentials,
            compression,
            wait_for_ready,
            timeout,
            metadata,
        )
