FROM jupyter/base-notebook:latest
USER root
RUN apt-get update && \
  apt-get -y upgrade && \
  pip install -r requirements.txt