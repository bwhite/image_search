import numpy as np
import distpy


class Base(object):

    def __init__(self, num_hash_bytes):
        super(Base, self).__init__()
        self._num_hash_bytes = num_hash_bytes

    def _check_store_hashes(self, hashes, ids):
        hashes = np.asarray(hashes, dtype=np.uint8)
        ids = np.asarray(ids, dtype=np.uint64)
        assert isinstance(hashes, np.ndarray)
        assert hashes.dtype == np.uint8
        assert hashes.ndim == 2
        assert hashes.shape[1] == self._num_hash_bytes
        assert hashes.shape[0] == ids.size
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.uint64
        assert ids.ndim == 1
        return hashes, ids

    def search_hash_nn(self, h):
        return self.search_hash_knn(h, 1)[0]

    def _check_search_hash_knn(self, h, k):
        h = np.asarray(h, dtype=np.uint8)
        assert isinstance(h, np.ndarray)
        assert h.dtype == np.uint8
        assert h.ndim == 1
        assert h.shape[0] == self._num_hash_bytes
        assert k > 0
        return h

    def search_hash_nn_multi(self, hs):
        for h in hs:
            yield self.search_hash_nn(h)

    def search_hash_knn_multi(self, hs, k):
        for h in hs:
            yield self.search_hash_knn(h, k)


class LinearHashDB(Base):

    def __init__(self, num_hash_bytes=None):
        super(LinearHashDB, self).__init__(num_hash_bytes)
        self._hashes = None
        self._ids = np.array([], dtype=np.uint64)
        self._d = distpy.Hamming()

    def store_hashes(self, hashes, ids):
        if self._num_hash_bytes is None:
            self._num_hash_bytes = hashes.shape[1]
        if self._hashes is None:
            self._hashes = np.zeros((0, self._num_hash_bytes), dtype=np.uint8)
        hashes, ids = self._check_store_hashes(hashes, ids)
        self._hashes = np.vstack([self._hashes, hashes])
        self._ids = np.hstack([self._ids, ids])
        return self

    def search_hash_knn(self, h, k):
        h = self._check_search_hash_knn(h, k)
        return self._ids[self._d.knn(self._hashes, h, k)[:, 1]]


class KDTreeHashes(Base):

    def __init__(self, num_hash_bytes):
        pass


class Classemes(Base):
    # http://vlg.cs.dartmouth.edu/projects/vlg_extractor/vlg_extractor/Home.html
    def __init__(self, num_hash_bytes):
        pass

class HammingEmbedding(Base):

    def __init__(self, num_hash_bytes):
        pass

class LocalitySensitiveHashing(Base):

    def __init__(self, num_hash_bytes):
        pass


class LinearFeatures(Base):

    def __init__(self, num_hash_bytes):
        pass

class ExemplarSVM(Base):

    def __init__(self, num_hash_bytes):
        pass

class MultipleViewHashing(Base):

    def __init__(self, num_hash_bytes):
        pass

# Top-K Ranking
# http://vlg.cs.dartmouth.edu/objclassretrieval/
# http://www.cs.dartmouth.edu/~csrs/posters/Chen_Fang.pdf


# Iterative Quantization: A Procrustean Approach to Learning Binary Codes
# http://www.cs.unc.edu/~lazebnik/publications/cvpr11_small_code.pdf

