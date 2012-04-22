from celery.task import task, subtask
import leveldb
import cPickle as pickle
from timer import timer
DB = None

@task
def get_image_kym(callback, **kw):
    global DB
    image_uri = kw['image_uri']
    if DB is None:
        DB = leveldb.LevelDB('/mnt/cassdrive0/crawer_machine_home_dirs/home/brandyn/crawl_cache')
    assert image_uri['type'] == 'KYM'
    with timer('get'):
        out = pickle.loads(DB.Get(image_uri['key']))
    return subtask(callback).delay(out, **kw)
