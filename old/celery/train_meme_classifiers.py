import cPickle as pickle
import imfeat
import sklearn.svm
import random
import numpy as np


meme_urls = pickle.load(open('/home/brandyn/kym/meme_photo_images.pkl'))
train_memes = set(x for x, y in meme_urls.items() if len(y) >= 50)
print(len(train_memes))
URL_FEATURES = []
fp = open('mapred/features.pkl')
while 1:
    try:
        URL_FEATURES.append(pickle.load(fp))
    except:
        break


def values_labels(meme_url, per_class=1000):
    pos_urls = meme_urls[meme_url]
    pos_label_values = [(1, y) for x, y in URL_FEATURES if x in pos_urls]
    neg_label_values = [(-1, y) for x, y in URL_FEATURES if x not in pos_urls]
    per_class = min([per_class, len(pos_label_values), len(neg_label_values)])
    if per_class < 50:
        raise ValueError
    random.shuffle(pos_label_values)
    random.shuffle(neg_label_values)
    label_values = pos_label_values[:per_class] + neg_label_values[:per_class]
    random.shuffle(label_values)
    labels, values = zip(*label_values)
    return np.array(values), np.array(labels)

for meme_url in train_memes:
    print(meme_url)
    try:
        c = sklearn.svm.LinearSVC().fit(*values_labels(meme_url))
    except ValueError:
        continue
    
