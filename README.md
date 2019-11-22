Python Docker 
===========
Start Python docker
-----

```
docker-compose run python
```

In container execute

```
python main.py
```


GPU
=========

Build Tensorflow Docker 
--------
```
docker build -t sebdei/tensorflow-gpu ./tensorflow
```

Run Docker with mounted volumes
----------
```
docker run --gpus all -it  \
--volume [absolute-path]/src:/src \
sebdei/tensorflow-gpu:latest
```

Check GPU activity
````
nvidia-smi -l 1
```

Jupyter
==========
Start docker
--------

```
docker-compose run --service-ports jupyter
```


