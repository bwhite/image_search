import scipy.io as sio
import pickle
import os
import glob
import gzip
import numpy as np
import scipy.misc.pilutil as smp

input_path = '/mnt/nfsdrives/shared/brandyn_deleteme/textons/run-1320116388.866126/view/'
output_path = '/mnt/nfsdrives/shared/behjat/output_data/sun09_sketch_queries/'
numClasses = 21
dim = 25

# GET THE NUMBER OF FILES
num_files = 0
for infile in glob.glob(os.path.join(input_path, '*.pkl.gz')):
    num_files = num_files + 1

# LOAD EACH FILE, RESIZE IT AND COPY IT INTO X
idx = 0
fList = glob.glob(os.path.join(input_path, '*.pkl.gz'))
fList.sort()
for infile in fList:
    temp = pickle.load(gzip.GzipFile(infile))
    masks = np.zeros(temp['all_probs2'].shape)
    image = temp['image']
    for k in temp['gt_masks']:
        if k in temp['classes']:
            idx = temp['classes'][k]
            masks[:, :, idx] = smp.imresize(temp['gt_masks'][k]/255, (temp['all_probs2'].shape[0], temp['all_probs2'].shape[1]), interp='nearest', mode='F')
    print os.path.join(output_path, infile[-12:])
    file = gzip.GzipFile(os.path.join(output_path, infile[-12:]), 'wb')
    data1 = {'all_probs2': masks, 'image': image}
    file.write(pickle.dumps(data1, 0))
    file.close()
    idx = idx+1
        
    #print os.path.join(output_path, infile[-12:-3])
    #output = open(os.path.join(output_path, infile[-12:-3]), 'wb')
    #data1 = {'all_probs2': masks, 'image': image}
    #pickle.dump(data1, output)
    #idx = idx+1
