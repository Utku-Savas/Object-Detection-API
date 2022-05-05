FROM nvcr.io/nvidia/cuda:11.5.1-cudnn8-runtime-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app


RUN apt-get update
RUN apt-get dist-upgrade --assume-yes
RUN apt-get install --assume-yes --no-install-recommends libgl1 libglib2.0-0 python3-minimal python3-pip

COPY requirements.txt .

RUN pip3 install --upgrade --no-cache -r requirements.txt

COPY yolo /app/yolo
COPY data /app/data

ENTRYPOINT ["/usr/bin/python3", "-m"]
CMD ["yolo"]
