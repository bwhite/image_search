import numpy as np
import distpy


class Base(object):

    def __init__(self, num_hash_bytes):
        super(Base, self).__init__()
        self._num_hash_bytes = num_hash_bytes

    def store(self, hashes, ids):
        assert isinstance(hashes, np.ndarray)
        assert hashes.dtype == np.uint8
        assert hashes.ndim == 2
        assert hashes.shape[1] == self._num_hash_bytes
        assert hashes.shape[0] == ids.size
        assert isinstance(ids, np.ndarray)
        assert ids.dtype == np.uint64
        assert ids.ndim == 1

    def search(self, h):
        assert isinstance(h, np.ndarray)
        assert h.dtype == np.uint8
        assert h.ndim == 1
        assert h.shape[0] == self._num_hash_bytes


class Linear(Base):

    def __init__(self, num_hash_bytes):
        super(Linear, self).__init__(num_hash_bytes)
        self._hashes = np.array([], dtype=np.uint8)
        self._ids = np.array([], dtype=np.uint64)
        self._d = distpy.Hamming(num_hash_bytes)

    def store(self, hashes, ids):
        super(Linear, self).store(hashes, ids)
        self._hashes = np.vstack([self._hashes, hashes])
        self._ids = np.vstack([self._ids, ids])

    def search(self, h, k=10):
        super(Linear, self).search(h)
        return self._ids[self._d.knn(self._hashes, h, k)[:, 1]]
