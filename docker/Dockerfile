FROM nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu18.04

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get update
RUN apt-get install -y python3-pip
RUN pip3 install -U pip
RUN apt-get update

COPY ../configs /app/configs

RUN pip install -U torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -U opennn