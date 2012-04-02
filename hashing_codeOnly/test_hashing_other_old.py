
def load_data(file_name):
    data = unpickle.unpickle(file_name)
    X = data['data']
    print 1
    return X
    #file_name = 'cifar-100-python/test'
    #coarse_labels = data['coarse_labels']
    #fine_labels = data['fine_labels']


def distMat(P1, P2):
    "Build a matrix of distances"
    X1 = np.tile(np.sum(np.square(P1), 1), (P2.shape[0], 1)).transpose()
    X2 = np.tile(np.sum(np.square(P2), 1), (P1.shape[0], 1)).transpose()
    R = np.dot(P1, P2.transpose())
    return np.sqrt(X1+X2.transpose()-2*R)

