from setuptools import setup

setup(
    name="face_restoration_worker_client",
    version="1.0.0",
    author="rm-rfred",
    packages=[
        "face_restoration_worker_client",
        "face_restoration_worker_client.face_restoration_worker_client",
        "face_restoration_worker_client.grpc_config",
    ],
    description="Grpc client for face restoration",
    install_requires=["protobuf==4.24.4", "grpcio==1.59.0", "grpcio-tools==1.59.0"],
)
