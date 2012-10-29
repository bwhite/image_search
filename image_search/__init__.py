import numpy as np
from db import LinearHashDB
from hashers import MedianHasher, RRMedianHasher, hash_bit_mean
from hik_hashers import HIKHasherGreedy


def _bool_to_hash(features):
    bit_shape = features.shape[0], int(np.ceil(features.shape[1] / 8.))
    return np.packbits(np.array(features, dtype=np.uint8)).reshape(bit_shape)
