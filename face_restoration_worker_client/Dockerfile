FROM nvcr.io/nvidia/pytorch:23.04-py3

WORKDIR /face_restoration_worker_client

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
COPY ./setup.py /setup.py
RUN python3.10 -m pip install -r /requirements.txt
