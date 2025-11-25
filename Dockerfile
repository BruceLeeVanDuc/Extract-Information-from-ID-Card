FROM python:3.8
WORKDIR /app
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
RUN set -xe && apt-get -yqq update && apt-get -yqq install python3-pip && pip3 install --upgrade pip
COPY requirements.txt /app
RUN pip install -r requirements.txt
COPY . /app
