import numpy as np
import cv2
import cPickle as pickle
import random
import kernels
import image_search


def resize_mask(mask, mask_size=32):
    mask = (mask).astype(np.float)
    return cv2.resize(mask, (mask_size, mask_size), interpolation=cv2.INTER_AREA)


def dist_spm(a, b):
    d = lambda x, y: kernels.histogram_intersection(x.reshape((1, -1)), y.reshape(1, -1)).flat[0]
    r = lambda x: resize_mask(x, x.shape[0] / 2)
    out = 0.
    weight = 1.
    while a.size != 1:
        out += d(a, b) * weight
        a, b = r(a), r(b)
        weight /= 4
    return out


def compute_mask_rect_area(mask, rect):
    rect = np.round(rect * np.hstack([mask.shape, mask.shape]))
    rect[0] = np.clip(rect[0], 0, mask.shape[0] - 2)
    rect[2] = np.clip(rect[2], 1, mask.shape[0] - 1) + 1
    if rect[2] == rect[0]:
        rect[0] -= 1
    rect[1] = np.clip(rect[1], 0, mask.shape[1] - 2)
    rect[3] = np.clip(rect[3], 1, mask.shape[1] - 1) + 1
    if rect[3] == rect[1]:
        rect[1] -= 1
    return np.mean(mask[rect[0]:rect[2], rect[1]:rect[3]])


def random_rect():
    tl = np.random.random(2)
    br = np.clip(tl + np.random.random(2), 0, 1)
    return np.hstack([tl, br])


class HIKHasherGreedy(object):

    def __init__(self, num_rects=100, num_queries=20, hash_bits=8):
        self.num_rects = num_rects
        self.num_queries = num_queries
        self.hash_bits = hash_bits

    def _image_masks_to_class_masks(self, image_masks):
        class_masks = {}  # [class_num] = masks
        for masks in image_masks:
            for class_num in range(masks.shape[2]):
                class_masks.setdefault(class_num, []).append(resize_mask(np.ascontiguousarray(masks[:, :, class_num])))
        return class_masks

    def __call__(self, masks):
        if masks.ndim == 3:
            masks = masks.resize(1, masks.shape[0], masks.shape[1], masks.shape[2])
        class_params = self.class_params.items()
        class_params.sort(key=lambda x: x[0])
        outs = []
        for mask in masks:
            out = []
            for class_num, params in class_params:
                class_mask = np.ascontiguousarray(resize_mask(mask[:, :, class_num]))
                for p in params['params']:
                    out.append(compute_mask_rect_area(class_mask, p['rect']) >= p['t'])
            outs.append(out)
        return image_search._bool_to_hash(np.array(out))

    def train(self, database_masks, query_masks=None):
        """
        Args:
            database_masks: Iter of masks
            query_masks: Iter of masks or None
        """
        database_class_masks = self._image_masks_to_class_masks(database_masks)
        if query_masks is None:
            query_class_masks = database_class_masks
        else:
            query_class_masks = self._image_masks_to_class_masks(query_masks)
        del query_masks
        del database_masks
        self.class_params = {}  # [class_num] = [(rect, thresh, weight)]
        for class_num, database_masks in database_class_masks.items():
            print(class_num)
            query_masks = query_class_masks[class_num]
            # Compute true SPM distance between all query and database masks
            query_dists = []
            for query_mask in query_masks[:self.num_queries]:
                query_dists.append([dist_spm(database_mask, query_mask)
                                    for database_mask in database_masks])
            query_dists = np.array(query_dists)
            # Greedily build up the hasher, each step learning a new bit
            prev_min_query_hash_bits = None
            prev_min_params = []
            total_residual = float('inf')
            total_weights = None
            for bit in range(self.hash_bits):
                # Randomly select rectangles
                min_params = ()
                min_query_hash_bits = None
                for rect_num in range(self.num_rects):
                    # Compute scalar values for each query and database mask
                    rect = random_rect()
                    t = random.random()
                    query_vals = np.array([compute_mask_rect_area(m, rect) for m in query_masks[:self.num_queries]]) >= t
                    database_vals = np.array([compute_mask_rect_area(m, rect) for m in database_masks]) >= t
                    single_query_hash_bits = np.dot(query_vals.reshape((-1, 1)), database_vals.reshape(1, -1))
                    single_query_hash_bits = single_query_hash_bits.reshape((single_query_hash_bits.shape[0],
                                                                             single_query_hash_bits.shape[1], 1))
                    if prev_min_query_hash_bits is None:
                        query_hash_bits = single_query_hash_bits
                    else:
                        query_hash_bits = np.dstack([prev_min_query_hash_bits, single_query_hash_bits])
                    coeffs, residuals, rank, sing_vals = np.linalg.lstsq(query_hash_bits.reshape((-1, query_hash_bits.shape[2])),
                                                                         query_dists.ravel())
                    try:
                        residual, = residuals
                    except ValueError:
                        continue
                    if total_residual > residual:
                        min_params = {'rect': rect, 't': t}
                        total_weights = coeffs
                        total_residual = residual
                        min_query_hash_bits = query_hash_bits
                    #score += len(query_rank.intersection(np.argsort(min_vals)[:1000]))
                    #print(score / len(database_vals))
                if min_params:
                    res = np.abs(np.dot(min_query_hash_bits, total_weights).ravel() - query_dists.ravel())
                    print('L1 Dist: m:%f med:%f mean:%f M:%f' % (np.min(res), np.median(res), np.mean(res), np.max(res)))
                    prev_min_query_hash_bits = min_query_hash_bits
                    prev_min_params.append(min_params)
            self.class_params[class_num] = {'params': prev_min_params, 'residual': total_residual, 'w': total_weights}
            pickle.dump(self.class_params, open('class_params.pkl', 'w'), -1)
        return self
