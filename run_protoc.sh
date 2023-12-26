#!/bin/bash
set -x # Echo on
python3.10 -m grpc_tools.protoc --proto_path=./face_restoration_worker_client/grpc_config --python_out=./face_restoration_worker_client/grpc_config --grpc_python_out=./face_restoration_worker_client/grpc_config ./face_restoration_worker_client/grpc_config/face_restoration.proto
