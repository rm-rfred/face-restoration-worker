# gRPC face-restoration-worker

A gRPC that exposes [CodeFormer](https://github.com/sczhou/CodeFormer/) model for face restoration
For additional ressources, you can check out the [paper](https://arxiv.org/abs/2206.11253)

![CodeFormer Architecture](./images/codeformer_architecture.jpg)

## :panda_face: Try Enhancing Old Photos / Fixing AI-arts

[<img src="images/imgsli_1.jpg" height="226px"/>](https://imgsli.com/MTI3NTE2) [<img src="images/imgsli_2.jpg" height="226px"/>](https://imgsli.com/MTI3NTE1) [<img src="images/imgsli_3.jpg" height="226px"/>](https://imgsli.com/MTI3NTIw)

#### Face Restoration

<img src="images/restoration_result1.png" width="400px"/> <img src="images/restoration_result2.png" width="400px"/>
<img src="images/restoration_result3.png" width="400px"/> <img src="images/restoration_result4.png" width="400px"/>

#### Face Color Enhancement and Restoration

<img src="images/color_enhancement_result1.png" width="400px"/> <img src="images/color_enhancement_result2.png" width="400px"/>

#### Face Inpainting

<img src="images/inpainting_result1.png" width="400px"/> <img src="images/inpainting_result2.png" width="400px"/>

## Run the project

```bash
git clone git@github.com:rm-rfred/face-restoration-worker.git
cd face-restoration-worker

# Download the models
bash download_models.sh

# Copy and fill the env file
cp .env.example .env

docker-compose build
docker-compose up -d
```

### Config files

face_restoration_pb2.py and face_restoration_pb2_grpc.py where generated by running :

```bash
bash run_protoc.sh
```

## Inference on GPU

In order to run inference on your GPU, you **must** have :

- NVIDIA driver installed
- NVIDIA container toolkit installed

Check out [here](https://github.com/NVIDIA/nvidia-container-toolkit) how to install it on your local device

Then, set DEVICE=cuda:0 on your .env file

## Why gRPC instead of REST ?

- Higher performances for microservice architecture
- High load APIs
- Better suited for real time / streaming apps

## gRPC architecture example

![gRPC](https://github.com/ByteByteGoHq/system-design-101/blob/main/images/grpc.jpg?raw=True)

## Dependencies

- Docker version **24.0.7**, build afdd53b
- Docker Compose version **v2.23.0**
- [NVIDIA container toolkit](https://github.com/NVIDIA/nvidia-container-toolkit)
