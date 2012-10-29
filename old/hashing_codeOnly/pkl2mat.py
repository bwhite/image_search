import scipy.io as sio
import pickle
import os
import glob
import gzip

input_path = '/mnt/nfsdrives/shared/brandyn_deleteme/textons/run-1320074371.422747/view/'
output_path = '/mnt/nfsdrives/shared/behjat/output_data/msrc_v30/training/'

for infile in glob.glob(os.path.join(input_path, '*.pkl.gz')):
    #temp = pickle.load(open(infile))
    temp = pickle.load(gzip.GzipFile(infile))
    fileName = output_path + infile[-12:-7] + '.mat'
    sio.savemat(fileName, temp)
