from timer import timer
import numpy as np
import cv2
import time

with timer('celery_import'):
    from celery.task import task, subtask
from importer import call_import
import data_worker

@task
def compute_image_feature(feature, **kw):
    with timer('get'):
        return data_worker.get_image_kym.delay(dict(compute_image_feature_bottom.subtask((feature,))), **kw)

@task
def compute_image_feature_bottom(image, feature, start_time, **kw):
    print('TotalTimeInitial[%f]' % (time.time() - start_time))
    with timer('fromstring'):
        image = cv2.imdecode(np.frombuffer(image, dtype=np.uint8), 1)
    with timer('import'):
        feature = call_import(feature)
    with timer('feature'):
        out = feature(image)
    print('TotalTime[%f]' % (time.time() - start_time))
    return out


@task
def learn_projection(hasher, features):
    # Load hasher
    # Compute projection matrix, mean, scale
    return x + y
