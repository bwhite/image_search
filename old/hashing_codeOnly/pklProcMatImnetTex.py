import scipy.io as sio
import pickle
import os
import glob
import gzip
import numpy as np
import scipy.misc.pilutil as smp

input_path = '/mnt/nfsdrives/shared/brandyn_deleteme/textons/run-1320218235.433508/view/'
output_path1 = '/mnt/nfsdrives/shared/behjat/output_data/imgNet_v01/baseData_25_texture.mat'
output_path2 = '/mnt/nfsdrives/shared/behjat/output_data/imgNet_v01/baseData_25_texture_pred.mat'
attribVocabulary = dict(wooden=0, furry=1, wet=2, smooth=3, metallic=4, vegetation=5, rough=6, shiny=7)
numClasses = 8
dim = 25

# ALLOCATE THE REQUIRED AMOUNT OF DATA
fList = glob.glob(os.path.join(input_path, '*.pkl.gz'))
num_files = fList.__len__()
X_gt = np.zeros((num_files, numClasses*dim*dim))

# GET THE CLASS INDICES
attribIdx = np.zeros((numClasses, 1))
infile = fList[0]
temp = pickle.load(gzip.GzipFile(infile))
idx = 0
for k in temp['classes']:
    if k in attribVocabulary:
        attribIdx[idx] = temp['classes'][k]
        idx = idx +1
attribIdx = np.int8(attribIdx.squeeze())

# LOAD EACH FILE, RESIZE IT AND COPY IT INTO X
fList.sort()
#idx = 0
#for infile in fList:
#    print idx
#    temp = pickle.load(gzip.GzipFile(infile))
#    tempMask = np.zeros((numClasses*dim*dim))
#    for k in temp['gt_masks']:
#        if k in attribVocabulary:
#            temp_pred = temp['gt_masks'][k]
#            temp_pred = temp_pred/255
#            temp_pred = smp.imresize(temp_pred, (dim, dim), interp='nearest', mode='F')
#            temp_pred = np.int8(temp_pred)
#            temp_pred = temp_pred.reshape(dim*dim, order='F').copy()
#            st = attribVocabulary[k]
#            tempMask[st::numClasses] = temp_pred
#    X_gt[idx, :] = tempMask
#    idx = idx+1

#X = {}
#X['X'] = X_gt
#sio.savemat(output_path1, X)

# ALLOCATE THE REQUIRED AMOUNT OF DATA
X_pred = np.zeros((num_files, numClasses*dim*dim))

# LOAD EACH FILE, RESIZE IT AND COPY IT INTO X
idx = 0
for infile in fList:
    print idx
    temp = pickle.load(gzip.GzipFile(infile))
    tempMask = np.zeros((numClasses*dim*dim))
    prob_map = temp['all_probs2']
    prob_map = prob_map[0:-1, 0:-1, attribIdx]
    prob_map = prob_map.argmax(2)
    prob_map = smp.imresize(prob_map, (dim, dim), interp='nearest', mode='F')
    prob_map = np.int8(prob_map)
    prob_map = prob_map.reshape(dim*dim, order='F').copy()
    for k in range(numClasses):
        tempClassMask = np.zeros((dim*dim, 1))
        tempClassMask[np.nonzero(prob_map==k)] = 1
        tempMask[k::numClasses] = tempClassMask.squeeze()
    X_pred[idx, :] = tempMask
    idx = idx+1

X = {}
X['X'] = X_pred
sio.savemat(output_path2, X)
