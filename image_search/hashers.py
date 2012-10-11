
import numpy as np


class MedianHasher(object):

    def __init__(self, remove_unused_features=True, center_features=True, normalize_features=True):
        self.used_inds = None
        self.sample_mean = None
        self.sample_std = None
        self.median_feature = None
        self.__center_features = center_features
        self.__normalize_features = normalize_features
        self.__remove_unused_features = remove_unused_features

    def __str__(self):
        return str(dict((x, getattr(self, x)) for x in ['used_inds', 'sample_mean', 'sample_std',
                                                        'median_feature']))

    def _bool_to_hash(self, features):
        bit_shape = features.shape[0], int(np.ceil(features.shape[1] / 8.))
        return np.packbits(np.array(features, dtype=np.uint8)).reshape(bit_shape)

    def _train_remove_unused_features(self, features):
        if features.shape[0] <= 1:
            self.used_inds = None
            print('Inds Unused[0] Used[%d]' % (features.shape[1]))
            return features
        self.used_inds = np.any(features[1:, :] != features[0, :], axis=0).nonzero()[0]
        print('Inds Unused[%d] Used[%d]' % (features.shape[1] - self.used_inds.size, self.used_inds.size))
        return self._remove_unused_features(features)

    def _train_center_features(self, features):
        self.sample_mean = np.mean(features, 0)
        return self._center_features(features)

    def _train_normalize_features(self, features):
        self.sample_std = np.std(features, 0)
        return self._normalize_features(features)

    def _remove_unused_features(self, features):
        if self.used_inds is None or self.used_inds.size == features.shape[1]:
            return features
        return np.ascontiguousarray(features[:, self.used_inds])

    def _center_features(self, features):
        return features - self.sample_mean

    def _normalize_features(self, features):
        return features / self.sample_std

    def _train_preprocess(self, features):
        features = self._condition_features(features, pad=False)
        if self.__remove_unused_features:
            features = self._train_remove_unused_features(features)
        if self.__center_features:
            features = self._train_center_features(features)
        if self.__normalize_features:
            features = self._train_normalize_features(features)
        return features

    def _train_postprocess(self, features):
        features = self._condition_features(features, pad=True)
        self.median_feature = np.median(features, 0)
        print('Hasher Median Train: NumDims[%d]' % (features.shape[1]))

    def train(self, features):
        self._train_postprocess(self._train_preprocess(features))
        return self

    def _condition_features(self, features, pad=True):
        features = np.asfarray(features)
        assert features.ndim < 3
        if features.ndim == 1:
            features = features.reshape((1, features.size))
        if pad and features.shape[1] % 8:
            features = np.hstack([features, np.zeros((features.shape[0], 8 - features.shape[1] % 8))])
        return np.ascontiguousarray(features, dtype=np.float64)

    def _preprocess(self, features):
        features = self._condition_features(features, pad=False)
        if self.__remove_unused_features:
            features = self._remove_unused_features(features)
        if self.__center_features:
            features = self._center_features(features)
        if self.__normalize_features:
            features = self._normalize_features(features)
        return features

    def _postprocess(self, features):
        features = self._condition_features(features, pad=True)
        return self._bool_to_hash(features > self.median_feature)

    def __call__(self, features):
        return self._postprocess(self._preprocess(features))


class RRMedianHasher(MedianHasher):

    def __init__(self, hash_bits=None, *args, **kw):
        assert hash_bits > 0 or hash_bits is None
        self.hash_bits = hash_bits
        super(RRMedianHasher, self).__init__(*args, **kw)

    def _train_postprocess(self, features):
        if self.hash_bits is None:
            self.hash_bits = features.shape[1]
        print('Hasher RR Train: NumDims[%d]' % (features.shape[1]))
        print('RR: Hashbits[%d]' % self.hash_bits)
        assert self.hash_bits <= features.shape[1]
        R = np.random.random((features.shape[1], self.hash_bits))
        U, S, V = np.linalg.svd(R)
        self.proj = U[:, :self.hash_bits]
        return super(RRMedianHasher, self)._train_postprocess(np.dot(features, self.proj))

    def _postprocess(self, features):
        return super(RRMedianHasher, self)._postprocess(np.dot(features, self.proj))


def hash_bit_mean(hashes):
    shape = hashes.shape[0], hashes.shape[1] * 8
    hash_bits = np.unpackbits(hashes).reshape(shape)
    return np.mean(hash_bits, 0)
