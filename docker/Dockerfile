FROM ubuntu:20.04

RUN apt-get update && apt-get install -y python3.9 python3-pip
RUN python3 -m pip install --upgrade pip

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip3 install opennn_pytorch
