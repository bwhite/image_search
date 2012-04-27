import leveldb
import cPickle as pickle
import os
import hadoopy
import picarus
import json


def data_iter():
    keys = set(sum([list(y) for x, y in pickle.load(open('/home/brandyn/kym/meme_photo_images.pkl')).items()], []))
    db = leveldb.LevelDB('/mnt/cassdrive0/crawer_machine_home_dirs/home/brandyn/crawl_cache')
    for num, (url, image_pkl) in enumerate(db.RangeIter()):
        if url not in keys or url.endswith('.gif'):
            continue
        image = pickle.loads(image_pkl)
        yield url, image


def mix_chain(*iterators):
    valid = True
    while valid:
        valid = False
        for i in iterators:
            try:
                v = i.next()
                if v is not None:
                    yield v
            except (OSError, StopIteration):
                pass
            else:
                valid = True


def main():
    f0 = {'name': 'imfeat.Histogram', 'args': ['lab']}
    f1 = {'name': 'imfeat.GIST'}
    feature_dict = {'name': 'imfeat.MetaFeature', 'args': [f0, f1], 'kw': {'max_side': 100}}
    iterators = []
    d = data_iter()
    orig_dir = os.path.abspath('.')
    for x in range(10):
        iterators.append(hadoopy.iterate_local(d, os.path.join(os.path.dirname(picarus.__file__),
                                                                         'vision', 'feature_compute.py'),
                                               cmdenvs=['FEATURE=%s' % json.dumps(feature_dict)],
                                               block_read=False))
    with open('features.pkl', 'w') as fp:
        for kv in mix_chain(*iterators):
            pickle.dump(kv, fp, -1)
    os.chdir(orig_dir)

main()
