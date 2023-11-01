FROM ubuntu:20.04

WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONPATH .

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository 'ppa:deadsnakes/ppa' 
RUN apt update && apt install -y wget python3.10 python3.10-dev

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

RUN rm get-pip.py

RUN python3.10 -m pip install --upgrade pip==23.3
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# gRPC healthcheck
RUN wget https://github.com/fullstorydev/grpcurl/releases/download/v1.8.2/grpcurl_1.8.2_linux_x86_64.tar.gz -O /grpcurl.tar.gz
RUN tar -xvzf /grpcurl.tar.gz
RUN chmod +x grpcurl && mv grpcurl /usr/local/bin/grpcurl && rm /grpcurl.tar.gz

COPY ./requirements.txt /app/requirements.txt

RUN python3.10 -m pip install -r /app/requirements.txt

COPY ./face_restoration_worker /app/face_restoration_worker/
COPY ./face_restoration_worker_client /app/face_restoration_worker_client/

CMD ["python3.10", "-m", "face_restoration_worker.server.grpc_server"]