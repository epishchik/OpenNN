FROM ubuntu:18.04

RUN apt-get update && apt-get install -y python3.7 python3-pip
RUN python -m pip install --upgrade pip

RUN pip install torch --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install opennn_pytorch