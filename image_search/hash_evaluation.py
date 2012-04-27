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


def samples_split_train_test(samples, num_train=1000):
    #Set Params
    num_test = samples.shape[0] - num_train

    # Randomize the samples
    random_index = range(samples.shape[0])
    random.shuffle(random_index)
    samples = np.ascontiguousarray(samples[random_index, :])

    # Split samples into train/test
    samples_test = samples[:num_test, :]
    samples_train = samples[num_test:, :]
    return samples_train, samples_test


def evaluate_hashing_method(samples_train, samples_test, bit, method):
    """

    Args:
        samples_train: samples x dims
        samples_test: samples x dims
        bit: Number of output bits
        method: Which method to use
    """
    num_training = samples_train.shape[0]
    print('NumTrain[%d] NumTest[%d]' % (len(samples_train), len(samples_test)))

    test_training_gt = make_ground_truth(samples_train, samples_test)

    #Generate training and test split and the data matrix
    XX = np.vstack((samples_train, samples_test))

    #Center the data
    sample_mean = np.mean(XX, 0)
    XX = XX - sample_mean

    #Evaluate different hashing approaches
    if method == 'ITQ':
        print 'ITQ'
        #Perform PCA
        a = np.cov(XX[:num_training, :], rowvar=0)
        D, V = np.linalg.eig(a)
        pc = V[:, np.argsort(D)[::-1][:bit]]
        XX = np.dot(XX, pc)
        #ITQ
        Y, proj = itq(XX[:num_training, :], 50)
        #XX = np.dot(XX, R)
        #Y = np.zeros(XX.shape)
        #Y[XX>=0] = 1
        #Y = np.packbits(np.array(Y > 0, dtype=np.uint8), 1)
    elif method == 'RR':
        print 'RR'
        #Perform PCA
        D, V = np.linalg.eig(np.cov(XX[:num_training, :], rowvar=0))
        pc = V[:, np.argsort(D)[::-1][:bit]]
        #RR
        R = np.random.rand(pc.shape[1], bit)
        U, S, V = np.linalg.svd(R)
        proj = np.dot(pc, U[:, :bit])
    else:
        raise ValueError('Unknown method [%s]' % method)
    #elif method == 'SKLSH':
    #    print 'SKLSH'
    #else:
    #    print 'LSH'
    #    XX = np.dot(XX, np.random.randn(XX.shape[1], bit))
    #    Y = np.zeros(XX.shape)
    #    Y[XX>=0]=1
    Y = hash_samples(XX, proj)
    print Y.shape
    B1 = Y[:num_training, :]
    B2 = Y[num_training:, :]
    Dhamm = HAMMING.cdist(B2, B1)
    precision, recall, rate = pr(test_training_gt, Dhamm)
    return {'precision': precision, 'recall': recall, 'rate': rate,
            'sample_mean': sample_mean, 'proj': proj}


def hash_samples(samples, proj, sample_mean=None):
    if sample_mean is not None:
        samples = samples - sample_mean
    return np.packbits(np.array(np.dot(samples, proj) >= 0, dtype=np.uint8), 1)


def itq(V, num_iter):
    bit = V.shape[1]
    R = np.random.rand(bit, bit)
    U, S, Vh = np.linalg.svd(R)
    R = U[:, :bit]

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


def pr(Wtrue, Dhat):
    print(Dhat.shape)
    print(Dhat.dtype)
    min_hamm = np.max([0, np.min(Dhat) - 1])
    max_hamm = np.max(Dhat)
    print((min_hamm, max_hamm))
    
    [Ntest, Ntrain] = Wtrue.shape
    total_good_pairs = np.sum(Wtrue)

    #Find pairs with similar codes
    num_hamm = max_hamm - min_hamm + 1
    precision = np.zeros(num_hamm)
    recall = np.zeros(num_hamm)
    rate = np.zeros(num_hamm)

    for m, n in enumerate(range(min_hamm, max_hamm)):
        j = (Dhat <= n)
        # Exp. num of good pairs that have exactly the same code
        retrieved_good_pairs = np.sum(Wtrue[j])
    	
        # Exp. num of total pairs that have exactly the same code
        retrieved_pairs = np.sum(j)

        precision[m] = retrieved_good_pairs / np.float(retrieved_pairs)
        recall[m] = retrieved_good_pairs / np.float(total_good_pairs)
        rate[m] = retrieved_pairs / np.float(Ntest*Ntrain)
    return precision, recall, rate


def print_gt_example(samples_train, samples_test, test_training_gt):
    print(samples_test[0, :])
    print(samples_train[test_training_gt[0], :])
    print('Mean Num neighbors[%f]' % (np.mean(np.sum(np.asfarray(test_training_gt), 0))))


def print_pr_ret(out):
    for x in np.vstack([out['precision'], out['recall'], out['rate']]).T:
        print x

if __name__ == '__main__':
    #train = np.random.random((1000, 16))
    #test = np.random.random((10000, 16))
    #test_training_gt = make_ground_truth(train, test)
    #samples = np.random.random((3000, 128))
    import pickle
    #samples = np.array(pickle.load(open('/home/brandyn/celery/features.pkl'))[:3000])
    samples = np.vstack(sum(pickle.load(open('../cifar_experiments/train.pkl')).values(), []) + sum(pickle.load(open('../cifar_experiments/test.pkl')).values(), []))
    #print(samples.shape)
    samples_train, samples_test = samples_split_train_test(samples)
    out = evaluate_hashing_method(samples_train, samples_test, 64, 'RR')
    print_pr_ret(out)
    #print_gt_example(train, test, test_training_gt)
