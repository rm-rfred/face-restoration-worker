FROM nvcr.io/nvidia/pytorch:23.04-py3

WORKDIR /

ENV DEBIAN_FRONTEND noninteractive
ENV PYTHONPATH .

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository 'ppa:deadsnakes/ppa' 
RUN apt update && apt install -y wget python3.10 python3.10-dev python3.10-distutils

RUN wget https://bootstrap.pypa.io/get-pip.py
RUN python3.10 get-pip.py

RUN rm get-pip.py

RUN python3.10 -m pip install --upgrade pip==23.3.2
RUN apt update && apt install -y libsm6 libxext6 libxrender-dev libgl1-mesa-glx

COPY ./requirements.txt /requirements.txt
COPY face_restoration_worker /face_restoration_worker
COPY face_restoration_worker_client /face_restoration_worker_client

RUN python3.10 -m pip install -r /requirements.txt
COPY ./setup.py /setup.py

COPY ./download_models.sh /download_models.sh
RUN bash /download_models.sh

CMD ["python3.10", "-m", "face_restoration_worker.face_restoration_server.grpc_server"]

HEALTHCHECK --interval=15s --timeout=30s --start-period=10s --retries=5 CMD [ "python3.10", "-m", "face_restoration_worker.face_restoration_server.grpc_healthcheck" ]