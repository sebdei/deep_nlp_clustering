Deep NLP Clustering implementation with an LSTM-Autoencoder
===============
Groundwork by [Xie et. al](https://arxiv.org/abs/1511.06335)

Pretrain an autoencoder, cut of the encoder part and use the latent features for clustering assignment hardening.


GPU
---------
[Tensorflow-GPU support with Docker](https://www.tensorflow.org/install/docker)

Build Tensorflow Docker 

```
docker build -t sebdei/tensorflow-gpu ./gpu
```

Run Docker with mounted volumes

```
docker run --gpus all -it  \
--volume ${PWD}/src:/src \
sebdei/tensorflow-gpu:latest
```

Check GPU activity
```
nvidia-smi -l 1
```


Python Docker 
------------
Start Python docker


```
docker-compose run python
```

Go directly to python console with `python`


Jupyter
--------
Start docker


```
docker-compose run --service-ports jupyter
```


