FROM "ubuntu:bionic"

MAINTAINER Suraj Pandey "suraj.pandey@trim-inc.com"

RUN apt update && \
    apt install -y python3-dev gcc

RUN mkdir -p /tensorflow/models/research

RUN apt-get install -y git python-pip
RUN pip install --upgrade pip

RUN pip install tensorflow==1.15

RUN apt-get install -y autoconf automake libtool curl make g++ unzip
RUN apt-get install -y protobuf-compiler python-pil python-lxml
RUN pip install jupyter
RUN pip install matplotlib

RUN git clone https://github.com/soorajpandey/image_detection.git /tensorflow/models/research

WORKDIR /tensorflow/models/research

RUN export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim

RUN cd "Flask Object Detection"

ADD requirements.txt requirements

ADD faster_rcnn_inception_v2_coco_2018_01_28 detection_image

ADD protos protos

ADD utils utils

ADD templates templates

ADD data data

ADD uploads uploads

RUN ls

RUN pip install -r requirements

EXPOSE 5000

CMD ["python", "app.py", "serve"]

