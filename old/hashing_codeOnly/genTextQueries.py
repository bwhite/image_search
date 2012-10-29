import scipy.io as sio
import pickle
import os
import glob
import gzip
import numpy as np
import scipy.misc.pilutil as smp

input_path = '/home/behjat/code/hashing/text_query_data/'
output_path = '/mnt/nfsdrives/shared/behjat/output_data/sun09_text_queries/'
numClasses = 21
dim = 25

query_types = {'lr_queries.mat', 'rr_queries.mat', 'nr_2obj_queries.mat', 'nr_3obj_queries.mat'}

for qType in query_types:
    data = sio.loadmat(os.path.join(input_path, qType))
    for i in range(data['X'].shape[0]):
        objects = data['objects'][0][i]
        relationships = data['relationships'][0][i]
        masks = np.zeros((dim, dim, numClasses))
        temp = data['X'][i]
        temp = np.asfarray(temp)
        for k in range(numClasses):
            temp_pred = temp[k::numClasses]
            temp_pred = temp_pred.reshape((dim, dim), order='F').copy()
            masks[:, :, k] = temp_pred
        fName = '%s_%05d.pkl.gz' % (qType[0:-4], i)
        print os.path.join(output_path, fName)
        file = gzip.GzipFile(os.path.join(output_path, fName), 'wb')
        data1 = {'all_probs2': masks, 'objects': objects, 'relationships': relationships}
        file.write(pickle.dumps(data1, 0))
        file.close()
                    
        #output = open(os.path.join(output_path, 'fName'), 'wb')
        #pickle.dump(data1, output)

