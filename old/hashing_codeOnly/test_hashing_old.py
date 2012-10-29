import unpickle
import numpy as np


def load_data(file_name):
    data = unpickle.unpickle(file_name)
    X = data['data']
    print 1
    return X
    #file_name = 'cifar-100-python/test'
    #coarse_labels = data['coarse_labels']
    #fine_labels = data['fine_labels']


def distMat(P1, P2):
    X1 = np.tile(np.sum(np.square(P1), 1), (P2.shape[0], 1)).transpose()
    X2 = np.tile(np.sum(np.square(P2), 1), (P1.shape[0], 1)).transpose()
    R = np.dot(P1, P2.transpose())
    return np.sqrt(X1+X2.transpose()-2*R)


def test(X, bit, method):
    #Set Params
    avgNumNeighbors = 50
    num_test = 1000
    bit = bit

    #Split into training and test data
    [nData, D] = X.shape
    #R = np.random.permutation(nData)
    R = np.arange(10000)
    Xtest = X[R[0:num_test], :]
    R = R[num_test:nData]
    Xtraining = X[R, :]
    num_training = Xtraining.shape[0]
    
    #Define Ground Truth Neighbors
    #R = np.random.permutation(num_training)
    R = np.arange(num_training)
    DtrueTraining = distMat(Xtraining[R[0:100], :], Xtraining)
    Dball = np.sort(DtrueTraining, 1)
    Dball = np.mean(Dball[:, avgNumNeighbors-1])
    print Dball
    
    #Scale data so that target distance is 1
    Xtraining = Xtraining/Dball
    Xtest = Xtest/Dball
    Dball = 1

    #Threshold to define Groundtruth
    DtrueTestTraining = distMat(Xtest, Xtraining)
    WtrueTestTraining = DtrueTestTraining < Dball

    #Generate training and test split and the data matrix
    XX = np.vstack((Xtraining, Xtest))

    #Center the data
    sampleMean = np.mean(XX, 0)
    XX = [XX - np.tile(sampleMean, (XX.shape[0], 1))]
    return XX
    
    #Evaluate different hashing approaches
    if method == 'ITQ':
        print 'ITQ'
        
    elif method == 'RR':
        print 'RR'

    elif method == 'SKLSH':
        print 'SKLSH'
        
    else:
        print 'LSH'
    

    
def hammingDist(B1, B2):
    bit_in_char = np.array([0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3,
    3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4,
    3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 1, 2,
    2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5,
    3, 4, 4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5,
    5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 1, 2, 2, 3,
    2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4,
    4, 5, 4, 5, 5, 6, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 2, 3, 3, 4, 3, 4,
    4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6,
    5, 6, 6, 7, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5,
    5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8], dtype='uint16')

    n1 = B1.shape[0]
    [n2, nwords] = B2.shape

    Dh = np.zeros([n1, n2], 'uint16')
    for j in range(n1):
        for n in range(nwords):
            y = np.bitwise_xor(B1[j, n], B2[:, n])
            Dh[j, :] = Dh[j, :] + bit_in_char[y]
    return Dh

