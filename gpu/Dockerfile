FROM tensorflow/tensorflow:nightly-gpu-py3
WORKDIR /src
COPY requirements.txt requirements.txt
RUN apt-get update && \
    apt-get -y upgrade && \
    apt-get -y install graphviz && \
    pip install -r requirements.txt
