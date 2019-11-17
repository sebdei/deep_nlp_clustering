For starting python docker 

```docker-compose run python```

For starting jupyter notebook

```docker-compose run --service-ports jupyter```

In container execute

```python main.py```

Clean docker rebuild

```
docker-compose rm -f
docker-compose up --build -d
````

