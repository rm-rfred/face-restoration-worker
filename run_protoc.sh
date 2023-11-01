#!/bin/bash
set -x # Echo on
python3.10 -m grpc_tools.protoc --proto_path=./face_restoration_worker_client/config --python_out=./face_restoration_worker_client/config --grpc_python_out=./face_restoration_worker_client/config ./face_restoration_worker_client/config/face_restoration.proto
