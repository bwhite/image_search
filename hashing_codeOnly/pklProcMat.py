import scipy.io as sio
import pickle
import os
import glob
import gzip
import numpy as np
import scipy.misc.pilutil as smp

input_path = '/mnt/nfsdrives/shared/brandyn_deleteme/textons/run-1320074371.422747/view/'
output_path = '/mnt/nfsdrives/shared/behjat/output_data/msrc_v30/baseData_25_pred_301.mat'
numClasses = 21
dim = 25
num_files = 0

# GET THE NUMBER OF FILES
for infile in glob.glob(os.path.join(input_path, '*.pkl.gz')):
    num_files = num_files + 1

# ALLOCATE THE REQUIRED AMOUNT OF DATA
X_pred_tr = np.zeros((num_files, numClasses*dim*dim))
X_gt_tr = np.zeros((num_files, numClasses*dim*dim))

# LOAD EACH FILE, RESIZE IT AND COPY IT INTO X
idx = 0
fList = glob.glob(os.path.join(input_path, '*.pkl.gz'))
fList.sort()
for infile in fList:
    print idx
    temp = pickle.load(gzip.GzipFile(infile))
    temp_pred = temp['max_classes2']
    temp_pred = temp_pred[0:-1, 0:-1]
    temp_pred = smp.imresize(temp_pred, (dim, dim), interp='nearest', mode='F')
    temp_pred = np.int8(temp_pred)
    temp_pred = temp_pred.reshape(dim*dim, order='F').copy()
    tempMask = np.zeros((numClasses*dim*dim))
    for k in range(numClasses):
        tempClassMask = np.zeros((dim*dim, 1))
        tempClassMask[np.nonzero(temp_pred==k)] = 1
        tempMask[k::numClasses] = tempClassMask.squeeze()
    X_pred_tr[idx, :] = tempMask
    idx = idx+1

idx = 0
for infile in fList:
    print idx
    temp = pickle.load(gzip.GzipFile(infile))
    tempMask = np.zeros((numClasses*dim*dim))
    for k in temp['gt_masks']:
        temp_pred = temp['gt_masks'][k]
        temp_pred = temp_pred/255
        temp_pred = smp.imresize(temp_pred, (dim, dim), interp='nearest', mode='F')
        temp_pred = np.int8(temp_pred)
        temp_pred = temp_pred.reshape(dim*dim, order='F').copy()
        st = temp['classes'][k]
        tempMask[st::numClasses] = temp_pred
    X_gt_tr[idx, :] = tempMask
    idx = idx+1

X = {}
X['X_pred_tr'] = X_pred_tr
X['X_gt_tr'] = X_gt_tr
sio.savemat(output_path, X)
