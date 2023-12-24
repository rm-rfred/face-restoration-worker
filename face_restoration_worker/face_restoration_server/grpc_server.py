from concurrent import futures
import pickle

import cv2
import grpc
import numpy as np
from simber import Logger

from face_restoration_worker.face_restoration_worker.face_restoration_inference import (
    inference,
)

LOG_FORMAT = "{levelname} [{filename}:{lineno}]:"
LOG_LEVEL: str = "INFO"
logger = Logger(__name__, log_path="/tmp/logs/server.log", level=LOG_LEVEL)
logger.update_format(LOG_FORMAT)

from face_restoration_worker_client.grpc_config import (
    face_restoration_pb2,
    face_restoration_pb2_grpc,
)


class FaceRestorationService(face_restoration_pb2_grpc.FaceRestorationServiceServicer):
    def ApplyFaceRestoration(self, request, context):
        try:
            img = np.array(request.image)
            img_rgb = cv2.cvtColor(pickle.loads(img), cv2.IMREAD_COLOR)

            restored_image_array = inference(
                img_rgb,
                request.background_enhance,
                request.face_upsample,
                request.upscale,
                request.codeformer_fidelity,
            )

            restored_image = pickle.dumps(restored_image_array)

            return face_restoration_pb2.FaceRestorationReply(
                restored_image=restored_image
            )
        except Exception as e:
            logger.error(e)


def serve():
    options = [
        ("grpc.max_send_message_length", -1),
        ("grpc.max_receive_message_length", -1),
        ("grpc.so_reuseport", 1),
        ("grpc.use_local_subchannel_pool", 1),
    ]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2), options=options)
    face_restoration_pb2_grpc.add_FaceRestorationServiceServicer_to_server(
        FaceRestorationService(), server
    )
    server.add_insecure_port("[::]:13000")
    logger.info("Binding to [::]:13000")
    server.start()
    server.wait_for_termination()
    logger.info("Server stopped")


if __name__ == "__main__":
    logger.info("Staring server...")
    serve()
