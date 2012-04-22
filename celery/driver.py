from timer import timer
with timer('import'):
    import vision_worker
import time
import cPickle as pickle
import random
keys = list(set(sum([list(y) for x, y in pickle.load(open('/home/brandyn/kym/meme_photo_images.pkl')).items()], [])))
random.shuffle(keys)

tasks = []


def get_result(task, levels):
    cur_tasks = [task]
    for i in range(levels - 1):
        cur_tasks.append(cur_tasks[-1].get())
    try:
        return cur_tasks[-1].get()
    finally:
        for t in cur_tasks:
            t.forget()

st = time.time()
for key in keys:
    with timer('Full'):
        feature = {'name': 'imfeat.Histogram', 'args': ['lab']}
        tasks.append(vision_worker.compute_image_feature.delay(feature, image_uri={'type': 'KYM', 'key': key}, start_time=time.time()))

features = []
for task in tasks:
    try:
        features.append(get_result(task, 3))
    except (ValueError, KeyError):
        pass
print(len(features))
print((time.time() - st) / len(features))
pickle.dump(features, open('features.pkl', 'w'), -1)
