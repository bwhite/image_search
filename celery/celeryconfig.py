# List of modules to import when celery starts.
CELERY_IMPORTS = ("vision_worker", "data_worker")

## Result store settings.
CELERY_RESULT_BACKEND = "redis"
CELERY_REDIS_HOST = "192.168.9.71"
CELERY_REDIS_PORT = 6379
CELERY_REDIS_DB = 0

## Broker settings.
BROKER_URL = "redis://192.168.9.71:6379/0"

## Worker settings
## If you're doing mostly I/O you can have more processes,
## but if mostly spending CPU, try to keep it close to the
## number of CPUs on your machine. If not set, the number of CPUs/cores
## available will be used.
CELERYD_POOL="processes"
CELERYD_CONCURRENCY = 16
CELERY_DISABLE_RATE_LIMITS=True
CELERY_ROUTES = {"tasks.url_chars": {"queue": "www"}, "data_worker.get_image_kym": {"queue": "kym_data"}}
CELERY_TASK_SERIALIZER="msgpack"

#CELERY_ANNOTATIONS = {"tasks.add": {"rate_limit": "10/s"}}
