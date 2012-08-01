import numpy as np
import distpy


class Base(object):

    def __init__(self, num_hash_bytes):
        super(Base, self).__init__()
        self._num_hash_bytes = num_hash_bytes

    def store_hashes(self, hashes, ids):
        assert isinstance(hashes, np.ndarray)
        assert hashes.dtype == np.uint8
        assert hashes.ndim == 2
        assert hashes.shape[1] == self._num_hash_bytes
        assert hashes.shape[0] == ids.size
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.uint64
        assert ids.ndim == 1

    def search_hash(self, h):
        assert isinstance(h, np.ndarray)
        assert h.dtype == np.uint8
        assert h.ndim == 1
        assert h.shape[0] == self._num_hash_bytes

    def search_hash_multi(self, hs, *args, **kw):
        for h in hs:
            yield self.search_hash(h, *args, **kw)


class LinearHashDB(Base):

    def __init__(self, num_hash_bytes):
        super(LinearHashDB, self).__init__(num_hash_bytes)
        self._hashes = np.array([], dtype=np.uint8).reshape((0, num_hash_bytes))
        self._ids = np.array([], dtype=np.uint64)
        self._d = distpy.Hamming()

    def store_hashes(self, hashes, ids):
        super(LinearHashDB, self).store_hashes(hashes, ids)
        self._hashes = np.vstack([self._hashes, hashes])
        self._ids = np.hstack([self._ids, ids])

    def search_hash(self, h, k=1):
        super(LinearHashDB, self).search_hash(h)
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

