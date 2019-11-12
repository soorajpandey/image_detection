FROM "ubuntu:bionic"
RUN apt-get update && yes | apt-get upgrade
RUN mkdir -p /tensorflow/models

RUN apt-get install -y git python-pip
RUN pip install --upgrade pip