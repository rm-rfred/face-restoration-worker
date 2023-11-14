FROM nvcr.io/nvidia/pytorch:23.04-py3

WORKDIR /app

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONPATH .

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository 'ppa:deadsnakes/ppa' 
RUN apt update && apt install -y wget python3.10 python3.10-dev python3.10-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

RUN rm get-pip.py

RUN python3.10 -m pip install --upgrade pip==23.3.1
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

# gRPC healthcheck
RUN wget https://github.com/fullstorydev/grpcurl/releases/download/v1.8.2/grpcurl_1.8.2_linux_x86_64.tar.gz -O /grpcurl.tar.gz
RUN tar -xvzf /grpcurl.tar.gz
RUN chmod +x grpcurl && mv grpcurl /usr/local/bin/grpcurl && rm /grpcurl.tar.gz

COPY ./requirements.txt /app/requirements.txt

# COPY ./setup.py /app/setup.py
# RUN python3.10 setup.py develop

# COPY ./code-fix.py /app/code-fix.py
# RUN python3.10 code-fix.py

# RUN python3.10 -m pip install basicsr
RUN python3.10 -m pip install -r /app/requirements.txt

COPY ./models /app/models
COPY ./face_restoration_worker /app/face_restoration_worker/
COPY ./face_restoration_worker_client /app/face_restoration_worker_client/

COPY ./blurry_face.jpg /app/blurry_face.jpg

CMD ["python3.10", "-m", "face_restoration_worker.server.grpc_server"]