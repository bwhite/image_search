import time
import contextlib


@contextlib.contextmanager
def timer(name):
    st = time.time()
    yield
    print('%s: %f' % (name, time.time() - st))
