import cPickle as pickle
import numpy as np
import image_search


def normalize_features(features):
    d = 8**3
    features = features.T
    features[:d, :] /= np.sum(features[:d, :], 0)
    features[d:, :] /= np.sum(features[d:, :], 0)
    return np.nan_to_num(features.T)
    

def main():
    image_uris, features = [], []
    with open('features.pkl') as fp:
        while 1:
            try:
                i, f = pickle.load(fp)
            except:
                break
            image_uris.append(i)
            features.append(f)
    print(len(image_uris))
    features = np.asfarray(features)
    features = normalize_features(features)
    train_samples, test_samples = image_search.samples_split_train_test(features)
    out = image_search.evaluate_hashing_method(train_samples, test_samples, 128, 'RR')
    print(image_search.print_pr_ret(out))
    print(features.shape)
    print(out['proj'].shape)
    print(out['sample_mean'].shape)
    hashes = image_search.hash_samples(features, out['proj'], out['sample_mean'])
    pickle.dump((image_uris, hashes, out), open('hashes.pkl', 'w'), -1)

if __name__ == '__main__':
    main()
    
