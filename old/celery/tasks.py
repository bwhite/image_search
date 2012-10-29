from celery.task import task
import requests

#(ignore_result=True)
@task
def add(x, y):
    return x + y

@task
def url_chars(x):
    return len(requests.get(x).content)
