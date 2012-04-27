import cPickle as pickle
import distpy
import gevent
from gevent import monkey
monkey.patch_all()
import bottle
import argparse
import cv2
from importer import call_import
import image_search
bottle.debug(True)
import numpy as np
import requests
from mapred.compute_db import normalize_features
IMAGE_KYM_URIS = HAMMING = ARGS = IMAGE_URIS = HASHES = HASH_DATA = FEATURE = None


@bottle.get(r'/<name:re:similar_meme_examples\.(js|html)>')
def similar_memes(name):
    ext = name.split('.')[1]
    url = bottle.request.query['url']
    print(url)
    f = FEATURE(cv2.imdecode(np.frombuffer(requests.get(url).content, dtype=np.uint8), 1))
    f = normalize_features(f.reshape((1, f.size)))
    h = image_search.hash_samples(f, HASH_DATA['proj'], HASH_DATA['sample_mean'])[0]
    similar = [(int(dist), IMAGE_URIS[ind], IMAGE_KYM_URIS[ind])
                       for dist, ind in HAMMING.knn(HASHES, h, 50)]
    meme_counts = {}
    for _, _, x in similar:
        try:
            meme_counts[x] += 1
        except KeyError:
            meme_counts[x] = 1
    meme_counts = sorted(meme_counts.items(), key=lambda x: x[1], reverse=True)
    out = {'similar': similar, 'meme_counts': meme_counts}
    if ext == 'js':
        return out
    return ''.join('<img src="%s"/>' % x for _, x, _ in out['similar'])
    

def main():
    global HAMMING, ARGS, IMAGE_URIS, HASHES, HASH_DATA, FEATURE, IMAGE_KYM_URIS
    parser = argparse.ArgumentParser(description="Serve a folder of images")
    # Webpy port
    parser.add_argument('--port', type=str, help='Run on this port (default 8080)',
                        default='8080')
    # These args are used as global variables
    ARGS = parser.parse_args()
    IMAGE_URIS, HASHES, HASH_DATA = pickle.load(open('mapred/hashes.pkl'))
    HAMMING = distpy.Hamming()
    f0 = {'name': 'imfeat.Histogram', 'args': ['lab']}
    f1 = {'name': 'imfeat.GIST'}
    feature_dict = {'name': 'imfeat.MetaFeature', 'args': [f0, f1], 'kw': {'max_side': 100}}
    #{'name': 'imfeat.Histogram', 'args': ['lab']})  # TODO(brandyn): Get this from hashes.pkl
    FEATURE = call_import(feature_dict)
    url_to_kym_url = {}
    for kym_url, urls in pickle.load(open('/home/brandyn/kym/meme_photo_images.pkl')).items():
        for url in urls:
            assert url not in url_to_kym_url  # Check if each is unique as we assume it
            url_to_kym_url[url] = kym_url
    IMAGE_KYM_URIS = [url_to_kym_url[image_uri] for image_uri in IMAGE_URIS]
        
    bottle.run(host='0.0.0.0', port=ARGS.port, server='gevent')


if __name__ == "__main__":
    main()
