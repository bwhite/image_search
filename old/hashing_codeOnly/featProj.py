import numpy as np
import scipy.misc.pilutil as smp
import scipy.io as sio
import hashingFunc as hl

# LOAD THE PARAMS
params_sun09 = sio.loadmat('sun09_params.mat')
params_imgNet_color = sio.loadmat('imgNet_color_params.mat')
params_imgNet_texture = sio.loadmat('imgNet_texture_params.mat')


# FUNCTION TO COMPUTE THE OBJECT HASHCODE
def hashObj(feat, params_sun09):
    dim = 25
    numClasses = 21

    # COMPUTE THE FEATURE REPRESENTATION
    temp_pred = feat['max_classes2']
    temp_pred = temp_pred[0:-1, 0:-1]
    temp_pred = smp.imresize(temp_pred, (dim, dim), interp='nearest', mode='F')
    temp_pred = np.int8(temp_pred)
    temp_pred = temp_pred.reshape(dim*dim, order='F').copy()
    tempMask = np.zeros((numClasses*dim*dim))
    for k in range(numClasses):
        tempClassMask = np.zeros((dim*dim, 1))
        tempClassMask[np.nonzero(temp_pred==k)] = 1
        tempMask[k::numClasses] = tempClassMask.squeeze()

    # COMPUTE THE HASHCODES FROM THE FEATURE
    tempMask = tempMask - np.tile(np.transpose(params_sun09['meanY']), (tempMask.shape[0], 1))
    tempMask = np.dot(tempMask, params_sun09['Wy'])
    tempMask = np.dot(tempMask, params_sun09['R'])
    Y = np.zeros(tempMask.shape)
    Y[tempMask>=0] = 1
    Z = hl.compactbit(Y>0)

    return Z


def hashAttrib(feat, params_imgNet_color, imgNet_texture):
    dim = 25
    numColorClasses = 11
    numTextureClasses = 8
    colorIdx = np.array([[0, 1, 2, 7, 10, 11, 13, 14, 15, 16, 18]], dtype='int8')
    textureIdx = np.array([[3, 4, 5, 6, 8, 12, 9, 17]], dtype='int8')
    
    # COMPUTE THE COLOR FEATURE REPRESENTATION
    tempMask = np.zeros((numColorClasses*dim*dim))
    prob_map = feat['all_probs2']
    prob_map = prob_map[0:-1, 0:-1, colorIdx]
    for k in range(numColorClasses):
        p1 = prob_map[:, :, k]
        p1 = smp.imresize(p1, (dim, dim), interp='bilinear', mode='F')
        p1 = p1.reshape(dim*dim, order='F').copy()
        tempMask[k::numColorClasses] = p1.squeeze()
                                
    # COMPUTE THE HASHCODES FROM THE FEATURE
    tempMask = tempMask - np.tile(np.transpose(params_imgNet_color['meanY']), (tempMask.shape[0], 1))
    tempMask = np.dot(tempMask, params_imgNet_color['Wy'])
    tempMask = np.dot(tempMask, params_imgNet_color['R'])
    Y = np.zeros(tempMask.shape)
    Y[tempMask>=0] = 1
    Zc = hl.compactbit(Y>0)

    # COMPUTE THE TEXTURE FEATURE REPRESENTATION
    tempMask = np.zeros((numTextureClasses*dim*dim))
    prob_map = feat['all_probs2']
    prob_map = prob_map[0:-1, 0:-1, textureIdx]
    for k in range(numTextureClasses):
        p1 = prob_map[:, :, k]
        p1 = smp.imresize(p1, (dim, dim), interp='bilinear', mode='F')
        p1 = p1.reshape(dim*dim, order='F').copy()
        tempMask[k::numTextureClasses] = p1.squeeze()

    # COMPUTE THE HASHCODES FROM THE FEATURE
    tempMask = tempMask - np.tile(np.transpose(params_imgNet_texture['meanY']), (tempMask.shape[0], 1))
    tempMask = np.dot(tempMask, params_imgNet_texture['Wy'])
    tempMask = np.dot(tempMask, params_imgNet_texture['R'])
    Y = np.zeros(tempMask.shape)
    Y[tempMask>=0] = 1
    Zt = hl.compactbit(Y>0)

    Z = np.hstack([Zc[0:127], Zt[0:127]])
    return Z
                                                                                    
