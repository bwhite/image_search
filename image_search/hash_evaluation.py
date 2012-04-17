import numpy as np
import scipy as sp
import scipy.spatial.distance
import distpy
import random

HAMMING = distpy.Hamming()


def make_ground_truth(samples_train, samples_test, avg_num_neighbors=50, thresh_sample_points=1000):
    """

    Args:
        samples_train: # training x dims
        samples_test: # testing x dims
        avg_num_neighbors: Number number used to determine average distance from true samples.
        thresh_sample_points: Number of training samples to use for learning the threshold.

    Returns:
        Boolean np array (# testing x # training) with true values corresponding to 'near'
        samples.
    """
    # Define Ground Truth Neighbors
    random_thresh_points = random.sample(xrange(samples_train.shape[0]), thresh_sample_points)
    training_sample_dist = sp.spatial.distance.cdist(samples_train[random_thresh_points, :], samples_train)
    mean_dist = np.mean(np.sort(training_sample_dist, 1)[:, avg_num_neighbors-1])

    # Threshold to define Groundtruth
    test_training_dist = sp.spatial.distance.cdist(samples_test, samples_train)
    test_training_gt = test_training_dist <= mean_dist
    print(test_training_gt.shape)
    return test_training_gt


def evaluate_hashing_methods(samples, bit, method):
    """

    Args:
        samples: samples x dims
        bit: Number of output bits
        method: Which method to use
    """
    #Set Params
    num_test = int(.5 * len(samples))

    # Randomize the samples
    random_index = range(samples.shape[0])
    random.shuffle(random_index)
    samples = np.ascontiguousarray(samples[random_index, :])

    # Split samples into train/test
    samples_test = samples[:num_test, :]
    samples_train = samples[num_test:, :]
    num_training = samples_train.shape[0]
    D = samples.shape[1]
    print('NumTrain[%d] NumTest[%d]' % (len(samples_train), len(samples_test)))

    test_training_gt = make_ground_truth(samples_train, samples_test)

    #Generate training and test split and the data matrix
    XX = np.vstack((samples_train, samples_test))

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
    Dhamm = HAMMING.cdist(B2, B1)
    return pr(test_training_gt, Dhamm)
    

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
        w = np.ceil(j / 8)
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


def pr(Wtrue, Dhat):
    max_hamm = np.max(Dhat)
    
    [Ntest, Ntrain] = Wtrue.shape
    total_good_pairs = np.sum(Wtrue)

    #Find pairs with similar codes
    precision = np.zeros(max_hamm)
    recall = np.zeros(max_hamm)
    rate = np.zeros(max_hamm)

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


def print_gt_example(samples_train, samples_test, test_training_gt):
    print(samples_test[0, :])
    print(samples_train[test_training_gt[0], :])
    print('Mean Num neighbors[%f]' % (np.mean(np.sum(np.asfarray(test_training_gt), 0))))

train = np.random.random((1000, 16))
test = np.random.random((10000, 16))
test_training_gt = make_ground_truth(train, test)

samples = np.random.random((10000, 16))
p, r, rate = evaluate_hashing_methods(samples, 16, 'ITQ')
print np.vstack([p, r]).T
#print_gt_example(train, test, test_training_gt)
