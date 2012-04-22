import numpy as np
import cPickle as pickle


def remove_zero_dims(features):
    keep_dims = np.nonzero(np.min(features, 0) != np.max(features, 0))[0]
    proj = np.zeros((features.shape[1], keep_dims.size))
    proj[keep_dims, np.arange(keep_dims.size)] = 1.
    return proj


def pca(data_matrix, tol=10**-10, centered=False, random=False):
    """Computes the Principle Component Analysis on a data_matrix.
                                                                                                                                            
    Args:
        data_matrix: Each row is a data point, each column is a feature for
            all points.  E.g., (points, feature_dims)
    Returns:
        A tuple (projection, mean) where
        projection: numpy array with shape (points, feature_dims)
        mean: numpy array with shape (feature_dims)
    """
    if centered:
        mean = np.zeros(data_matrix.shape[1])
    else:
        mean = np.mean(data_matrix, 0)
        data_matrix = data_matrix - mean
    U, S, V = np.linalg.svd(data_matrix, full_matrices=False)
    V = V[S > tol, :].T
    np.random.random(V.shape[1])
    if random:
        random_rotation = np.linalg.qr(np.random.random((V.shape[1], V.shape[1])))[0]
        return np.dot(V, random_rotation), mean
    return V, mean


def pca_proj(data_matrix, *args, **kw):
    mean = np.mean(data_matrix, 0)
    data_matrix = data_matrix - mean
    v, _ = pca(data_matrix, centered=True, *args, **kw)
    return np.dot(data_matrix, v), v, mean


def remove_zero_dims_proj(data_matrix, *args, **kw):
    proj = remove_zero_dims(data_matrix, *args, **kw)
    return np.dot(data_matrix, proj)


data_matrix = np.array(pickle.load(open('features.pkl')))[:1000, :]
print(data_matrix.shape)
print(pca_proj(data_matrix)[0][0, :].shape)
print(pca_proj(data_matrix, random=True)[0][0, :].shape)
print(remove_zero_dims_proj(data_matrix)[0, :].shape)
#p = remove_zero_dims(arr)
#print p.shape
#arr2 = np.dot(arr, p)
#print arr[0][:64]
#print pca(arr)[0].shape
#arr2[0][:64]
