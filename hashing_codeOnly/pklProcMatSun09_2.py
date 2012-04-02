import scipy.io as sio
import pickle
import os
import glob
import gzip
import numpy as np
import scipy.misc.pilutil as smp

input_path1 = '/mnt/nfsdrives/shared/brandyn_deleteme/textons/run-1320116333.293902/view/'
input_path2 = '/mnt/nfsdrives/shared/brandyn_deleteme/textons/run-1320116388.866126/view/'
output_path1 = '/mnt/nfsdrives/shared/behjat/output_data/sun09_v01/baseData_25_prob_pred1.mat'
output_path2 = '/mnt/nfsdrives/shared/behjat/output_data/sun09_v01/baseData_25_prob_pred2.mat'
numClasses = 21
dim = 25

# GET THE NUMBER OF FILES
num_files = 0
for infile in glob.glob(os.path.join(input_path1, '*.pkl.gz')):
    num_files = num_files + 1

# ALLOCATE THE REQUIRED AMOUNT OF DATA
X_pred_tr = np.zeros((num_files, numClasses*dim*dim))

# LOAD EACH FILE, RESIZE IT AND COPY IT INTO X
idx = 0
fList = glob.glob(os.path.join(input_path1, '*.pkl.gz'))
fList.sort()
for infile in fList:
    print idx
    temp = pickle.load(gzip.GzipFile(infile))
    prob_map = temp['all_probs2']
    prob_map = prob_map[0:-1, 0:-1]
    tempMask = np.zeros((numClasses*dim*dim))
    for k in range(numClasses):
        p1 = prob_map[:, :, k]
        p1 = smp.imresize(p1, (dim, dim), interp='bilinear', mode='F')
        p1 = p1.reshape(dim*dim, order='F').copy()
        tempMask[k::numClasses] = p1.squeeze()
    X_pred_tr[idx, :] = tempMask
    idx = idx+1

X = {}
X['X'] = X_pred_tr
sio.savemat(output_path1, X)

# GET THE NUMBER OF FILES
num_files = 0
for infile in glob.glob(os.path.join(input_path2, '*.pkl.gz')):
    num_files = num_files + 1

# ALLOCATE THE REQUIRED AMOUNT OF DATA
X_pred_te = np.zeros((num_files, numClasses*dim*dim))

# LOAD EACH FILE, RESIZE IT AND COPY IT INTO X
idx = 0
fList = glob.glob(os.path.join(input_path2, '*.pkl.gz'))
fList.sort()
for infile in fList:
    print idx
    temp = pickle.load(gzip.GzipFile(infile))
    prob_map = temp['all_probs2']
    prob_map = prob_map[0:-1, 0:-1]
    tempMask = np.zeros((numClasses*dim*dim))
    for k in range(numClasses):
        p1 = prob_map[:, :, k]
        p1 = smp.imresize(p1, (dim, dim), interp='bilinear', mode='F')
        p1 = p1.reshape(dim*dim, order='F').copy()
        tempMask[k::numClasses] = p1.squeeze()
    X_pred_te[idx, :] = tempMask
    idx = idx+1
    
X = {}
X['X'] = X_pred_te
sio.savemat(output_path2, X)
