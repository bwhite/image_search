import numpy as np
import scipy as sp
import scipy.spatial.distance


def test(X, bit, method):
    """

    Args:
        X: samples x dims
        bit: Number of output bits
        method: Which method to use
    """
    #Set Params
    avgNumNeighbors = 50
    num_test = int(.5 * len(X))

    #Split into training and test data
    nData, D = X.shape
    Xtest = X[:num_test, :]
    Xtraining = X[num_test:, :]
    num_training = Xtraining.shape[0]
    print('NumTrain[%d] NumTest[%d]' % (len(Xtraining), len(Xtest)))

    #Define Ground Truth Neighbors
    DtrueTraining = sp.spatial.distance.cdist(Xtraining[:100, :], Xtraining)  # NOTE(brandyn): used 100 values (constant) before
    print(DtrueTraining)
    Dball = np.sort(DtrueTraining, 1)
    Dball = np.mean(Dball[:, avgNumNeighbors-1])
    print 'Dball[%s]' % Dball  # ?: Why?

    #Scale data so that target distance is 1
    Xtraining = Xtraining / Dball
    Xtest = Xtest / Dball

    #Threshold to define Groundtruth
    DtrueTestTraining = sp.spatial.distance.cdist(Xtest, Xtraining)
    WtrueTestTraining = DtrueTestTraining < 1
    print('NumTrue[%f]' % (np.sum(np.asfarray(WtrueTestTraining)) / WtrueTestTraining.size))

    #Generate training and test split and the data matrix
    XX = np.vstack((Xtraining, Xtest))

    #Center the data
    sampleMean = np.mean(XX, 0)
    XX = XX - sampleMean

    #Evaluate different hashing approaches
    if method == 'ITQ':
        print 'ITQ'
        #Perform PCA
        a = np.cov(XX[0:num_training, :], rowvar=0)
        D, V = np.linalg.eig(a)
        pc = V[:, 0:32]
        XX = np.dot(XX, pc)
        #ITQ
        Y, R = itq(XX[0:num_training, :], 50)
        XX = np.dot(XX, R)
        Y = np.zeros(XX.shape)
        Y[XX>=0] = 1
        Y = compactbit(Y>0)
		
    elif method == 'RR':
        print 'RR'
        #Perform PCA
        a = np.cov(XX[0:num_training, :], rowvar=0)
        D, V = np.linalg.eig(a)
        pc = V[:, 0:32]
        XX = np.dot(XX, pc)
        #RR
        R = np.random.rand(XX.shape[1], bit)
        U, S, V = np.linalg.svd(R)
        XX = np.dot(XX, U[:, :bit])

    elif method == 'SKLSH':
        print 'SKLSH'

    else:
        print 'LSH'
        XX = np.dot(XX, np.random.randn(XX.shape[1], bit))
        Y = np.zeros(XX.shape)
        Y[XX>=0]=1
       
    B1 = Y[:num_training, :]
    B2 = Y[num_training:, :]
    import distpy
    print(B1.dtype)
    import time
    st = time.time()
    Dhamm2 = distpy.Hamming().cdist(B2, B1)
    print(time.time() - st)
    st = time.time()
    #Dhamm = hammingDist(B2, B1)
    #print(time.time() - st)
    #np.testing.assert_equal(Dhamm[0], Dhamm2[0])
    #np.testing.assert_equal(Dhamm, Dhamm2)
    print(Dhamm2[0])
    #print(Dhamm[0])
    return pr(WtrueTestTraining, Dhamm2)
    

def itq(V, num_iter):
    bit = V.shape[1]
    R = np.random.rand(bit, bit)
    U, S, Vh = np.linalg.svd(R)
    R = U[:, 0:bit]

    for iter in range(num_iter+1):
        Z = np.dot(V, R)
        UX = np.ones(Z.shape)*-1
        UX[Z>=0] = 1
        C = np.dot(UX.transpose(), V)
        UB, sigma, UA = np.linalg.svd(C)
        R = np.dot(UA, UB.transpose())

    #make B binary
    B = UX
    B[B<0] = 0
    return B, R


def compactbit(b):
    [nSamples, nbits] = b.shape
    nwords = np.ceil(nbits/8)
    cb = np.zeros([nSamples, nwords], dtype='uint8')

    for j in range(np.int(nwords)):
        w = np.ceil(j/8)
        for k in range(nSamples):
            s = j*8
            e = min((j+1)*8, nbits)
            cb[k, j] = bin2dec(b[k, s:e])
    return cb


def bin2dec(b):
    d = 0
    for i in range(len(b)):
        d = d + 2**i*b[i]
    return d


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


def pr(Wtrue, Dhat):
    max_hamm = np.max(Dhat)
    
    [Ntest, Ntrain] = Wtrue.shape
    total_good_pairs = np.sum(Wtrue)

    #Find pairs with similar codes
    precision = np.zeros([max_hamm, 1])
    recall = np.zeros([max_hamm, 1])
    rate = np.zeros([max_hamm, 1])

    for n in range(len(precision)):
        j = (Dhat<=(n+0.00001))
        # Exp. num of good pairs that have exactly the same code
        retrieved_good_pairs = np.sum(Wtrue[j])
    	
        # Exp. num of total pairs that have exactly the same code
        retrieved_pairs = np.sum(j)

        precision[n] = retrieved_good_pairs / np.float(retrieved_pairs)
        recall[n] = retrieved_good_pairs / np.float(total_good_pairs)
        rate[n] = retrieved_pairs / np.float(Ntest*Ntrain)
    return precision, recall, rate

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
    5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 7], dtype='uint16')

recall, precision, rate = test(np.random.random((3000, 128)), 256, 'ITQ')
