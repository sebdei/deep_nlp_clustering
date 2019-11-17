FROM jupyter/base-notebook:latest
USER root
COPY requirements.txt requirements.txt
RUN apt-get update && \
  apt-get -y upgrade && \
  pip install -r requirements.txt