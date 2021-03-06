from compute_hashes import compute_hash
from build_features import compute_feature
import numpy as np
import cv2
import cPickle as pickle
import image_search


def evaluate_image(image, db, image_metadatas, median_feature, k=50, max_tags=10):
    feature = compute_feature(image)
    feature = feature.reshape((1, feature.size))
    h = compute_hash(feature, median_feature).ravel()
    # Compute K-nn
    tags = []
    with open('results.html', 'w') as fp:
        fp.write('<img height=250 src=%s \><br>' % 'target.jpg')
        for index in db.search(h, k=k):
            tags += image_metadatas[index]['tags']
            fp.write('<img height=150 src=%s \>' % image_metadatas[index]['url_m'])
    # Pool tags
    tag_hist = {}
    for tag in tags:
        try:
            tag_hist[tag] += 1
        except KeyError:
            tag_hist[tag] = 1
    return sorted(tag_hist.items(), reverse=True, key=lambda x: x[1])[:max_tags]


def main():
    with open('database.pkl') as fp:
        hashes, image_metadatas, median_feature = pickle.load(fp)
    hashes = np.array([x.ravel() for x in hashes])
    db = image_search.db.Linear(hashes.shape[1])
    db.store(hashes, np.array(np.arange(hashes.shape[0]), dtype=np.uint64))
    for x in image_metadatas:
        x['tags'] = x['tags'].split()
    print('Model loaded')
    # load test image
    image = cv2.imread('target.jpg')
    print(evaluate_image(image, db, image_metadatas, median_feature))

main()
