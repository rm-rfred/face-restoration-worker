import grpc

from face_restoration_worker_client.grpc_config import (
    face_restoration_pb2,
    face_restoration_pb2_grpc,
)


class GrpcClient(object):
    @staticmethod
    def get_face_restoration_from_grpc(endpoint: str, image: bytes, timeout: int = 60):
        return GrpcClient.face_restoration(
            endpoint=endpoint, image=image, timeout=timeout
        )

    @staticmethod
    def face_restoration(endpoint: str, image, timeout: int = 60):
        """Apply face restoration on image face

        Arguments:
            endpoint (str): Server endpoint
            image: The image to apply face restoration
            timeout (int): Maximum seconds to process an image
        """
        channel = grpc.insecure_channel(
            endpoint,
            options=[
                ("grpc.max_send_message_length", -1),
                ("grpc.max_receive_message_length", -1),
                ("grpc.so_reuseport", 1),
                ("grpc.use_local_subchannel_pool", 1),
            ],
        )
        stub = face_restoration_pb2_grpc.FaceRestorationServiceStub(channel)

        response = stub.ApplyFaceRestoration(
            face_restoration_pb2.FaceRestorationRequest(
                image=image,
            ),
            timeout=timeout,
        )
        return response.restored_image
