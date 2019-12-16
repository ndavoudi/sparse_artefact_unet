

import sklearn
from sklearn import metrics
import sys
import glob 
import re
import os
import time
import numpy as np
import lasagne
from lasagne import layers
import theano
import theano.tensor as T
floatX = theano.config.floatX
import random
import h5py 	
import matplotlib.pyplot as plt


import scipy.io as sio

from skimage.measure import compare_ssim as ssim


class Unpool2DLayer(layers.Layer):
    """
    This layer performs unpooling over the last two dimensions
    of a 4D tensor.
    """
    def __init__(self, incoming, ds, **kwargs):

        super(Unpool2DLayer, self).__init__(incoming, **kwargs)

        if (isinstance(ds, int)):
            ds = (ds, ds)
        else:
            ds = tuple(ds)
            if len(ds) != 2:
                raise ValueError('ds must be an int or pair of int')
        self.ds = ds

    def get_output_shape_for(self, input_shape):
        output_shape = list(input_shape)

        output_shape[2] = input_shape[2] * self.ds[0]
        output_shape[3] = input_shape[3] * self.ds[1]

        return tuple(output_shape)

    def get_output_for(self, input, **kwargs):
        ds = self.ds
        input_shape = input.shape
        output_shape = self.get_output_shape_for(input_shape)
        return input.repeat(ds[0], axis=2).repeat(ds[1], axis=3)




def signaltonoise(a, axis=0, ddof=0):
    """
    The signal-to-noise ratio of the input data.
    Returns the signal-to-noise ratio of `a`, here defined as the mean
    divided by the standard deviation.
    Parameters
    ----------
    a : array_like
        An array_like object containing the sample data.
    axis : int or None, optional
        If axis is equal to None, the array is first ravel'd. If axis is an
        integer, this is the axis over which to operate. Default is 0.
    ddof : int, optional
        Degrees of freedom correction for standard deviation. Default is 0.
    Returns
    -------
    s2n : ndarray
        The mean to standard deviation ratio(s) along `axis`, or 0 where the
        standard deviation is 0.
    """
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return np.where(sd == 0, 0, m/sd)


def calc_psnr(GT_img, recon_img):

    mse = sklearn.metrics.mean_squared_error(np.squeeze(GT_img, axis = 0), np.squeeze(recon_img,axis = 0))
    return 10*np.log10(np.max(np.power(GT_img,2))/mse)
    





def dice(im1, im2):
    """
    Computes the Dice coefficient, a measure of set similarity.
    Parameters
    ----------
    im1 : array-like, bool
        Any array of arbitrary size. If not boolean, will be converted.
    im2 : array-like, bool
        Any other array of identical size. If not boolean, will be converted.
    Returns
    -------
    dice : float
        Dice coefficient as a float on range [0,1].
        Maximum similarity = 1   
        No similarity = 0
        
    Notes
    -----
    The order of inputs for `dice` is irrelevant. The result will be
    identical if `im1` and `im2` are switched.
    """

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / (im1.sum() + im2.sum())




def augmentation(image, imageB, org_width=160,org_height=224, width=190, height=262):
    max_angle=20
    image=cv2.resize(image,(height,width))
    imageB=cv2.resize(imageB,(height,width))

    angle=np.random.randint(max_angle)
    if np.random.randint(2):
        angle=-angle
    image=rotate(image,angle,resize=True)
    imageB=rotate(imageB,angle,resize=True)

    xstart=np.random.randint(width-org_width)
    ystart=np.random.randint(height-org_height)
    image=image[xstart:xstart+org_width,ystart:ystart+org_height]
    imageB=imageB[xstart:xstart+org_width,ystart:ystart+org_height]

    if np.random.randint(2):
        image=cv2.flip(image,1)
        imageB=cv2.flip(imageB,1)
    
    if np.random.randint(2):
        image=cv2.flip(image,0)
        imageB=cv2.flip(imageB,0)

    image=cv2.resize(image,(org_height,org_width))
    imageB=cv2.resize(imageB,(org_height,org_width))

    return image,imageB


def lasagne2str(network):

    net_layers = lasagne.layers.get_all_layers(network)
    result = ''
    for layer in net_layers:
        t = type(layer)
        if t is lasagne.layers.input.InputLayer:
            pass
        elif t is lasagne.layers.conv.Conv2DLayer:
            result += '{}[{}] '.format(layer.num_filters, 'x'.join([str(fs) for fs in layer.filter_size]))
        elif t is lasagne.layers.DropoutLayer:
            result += 'drop:{:g} '.format(layer.p)
        elif t is voxnet.layers.Conv3dMMLayer:
            result+= '{}[{}] '.format(layer.num_filters, 'x'.join([str(fs) for fs in layer.filter_size]))
        elif t is voxnet.layers.MaxPool3dLayer:
            result+= 'max[{}] '.format('x'.join([str(ps) for ps in layer.pool_shape]))
        elif t is lasagne.layers.DenseLayer:
            result+= 'Dense:{:g} '.format(layer.num_units)
        else:
            result += t.__name__ +' '
        result += str(lasagne.layers.get_output_shape(layer, input_shapes=None))+' '
    return result.strip()





def normalize_data(X, axis = 0):
    mu = np.mean(X,axis =0)
    sigma = np.std(X, axis=0) 
        
    X_norm = 2*(X-mu)/sigma
    return np.nan_to_num(X_norm)



def split_train_test(idc, test_fraction):

    random.seed(0) # deterministic
    random.shuffle(idc) # in-place
    tmp = int((1-test_fraction)*len(idc))
    idc_train = idc[:tmp] # this makes a copy
    idc_test = idc[tmp:] 
    return idc_train, idc_test



def get_train_fn(learning_rate):
    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate) 
    train_fn = theano.function([tX, tY], [tYhat, loss], updates=updates, on_unused_input='ignore')
    return train_fn



tX = T.tensor4('inputs') 
tY = T.tensor3	('targets') 
minibatch_size = 1


inputLayer = lasagne.layers.InputLayer(shape=(None, 1, 512, 512), input_var=tX)

conv1 = lasagne.layers.Conv2DLayer(inputLayer, num_filters= 64, filter_size=(3,3), stride=1, pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv1 = lasagne.layers.Conv2DLayer(conv1, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis1 = lasagne.layers.get_output(conv1)

pool1 = lasagne.layers.Pool2DLayer(conv1, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')

vis2 = lasagne.layers.get_output(pool1)

conv2 = lasagne.layers.Conv2DLayer(pool1, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv2 = lasagne.layers.Conv2DLayer(conv2, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)
 
vis3 = lasagne.layers.get_output(conv2)

pool2 = lasagne.layers.Pool2DLayer(conv2, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')

vis4 = lasagne.layers.get_output(pool2)

conv3 = lasagne.layers.Conv2DLayer(pool2, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv3 = lasagne.layers.Conv2DLayer(conv3, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis5 = lasagne.layers.get_output(conv3)

pool3 = lasagne.layers.Pool2DLayer(conv3, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')

vis6 = lasagne.layers.get_output(pool3)

conv4 = lasagne.layers.Conv2DLayer(pool3, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv4 = lasagne.layers.Conv2DLayer(conv4, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis7 = lasagne.layers.get_output(conv4)

pool4 = lasagne.layers.Pool2DLayer(conv4, pool_size = (2,2), stride=None, pad=(0, 0), ignore_border=True, mode='average_inc_pad')

vis8 = lasagne.layers.get_output(pool4)

conv5 = lasagne.layers.Conv2DLayer(pool4, num_filters= 1024, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv5 = lasagne.layers.Conv2DLayer(conv5, num_filters= 1024, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis9 = lasagne.layers.get_output(conv5)

up6 = Unpool2DLayer(conv5, (2,2))

vis10 = lasagne.layers.get_output(up6)

conv6 = lasagne.layers.Conv2DLayer(up6, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis11 = lasagne.layers.get_output(conv6)

merge6 = lasagne.layers.ConcatLayer([conv6, conv4], axis = 1)

vis12 = lasagne.layers.get_output(merge6)

conv6 = lasagne.layers.Conv2DLayer(merge6, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv6 = lasagne.layers.Conv2DLayer(conv6, num_filters= 512, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis13 = lasagne.layers.get_output(conv6)

up7 = Unpool2DLayer(conv6, (2,2))

vis14 = lasagne.layers.get_output(up7)

conv7 = lasagne.layers.Conv2DLayer(up7, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis15 = lasagne.layers.get_output(conv7)

merge7 = lasagne.layers.ConcatLayer([conv7, conv3], axis = 1)

vis16 = lasagne.layers.get_output(merge7)

conv7 = lasagne.layers.Conv2DLayer(merge7, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv7 = lasagne.layers.Conv2DLayer(conv7, num_filters= 256, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis17 = lasagne.layers.get_output(conv7)

up8 = Unpool2DLayer(conv7, (2,2))

vis18 = lasagne.layers.get_output(up8)

conv8 = lasagne.layers.Conv2DLayer(up8, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis19 = lasagne.layers.get_output(conv8)

merge8 = lasagne.layers.ConcatLayer([conv8, conv2], axis = 1)

vis20 = lasagne.layers.get_output(merge8)

conv8 = lasagne.layers.Conv2DLayer(merge8, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

conv8 = lasagne.layers.Conv2DLayer(conv8, num_filters= 128, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis21 = lasagne.layers.get_output(conv8)

up9 = Unpool2DLayer(conv8, (2,2))

vis22 = lasagne.layers.get_output(up9)

conv9 = lasagne.layers.Conv2DLayer(up9, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis23 = lasagne.layers.get_output(conv9)

merge9 = lasagne.layers.ConcatLayer([conv9, conv1], axis = 1)

vis24 = lasagne.layers.get_output(merge9)

conv9 = lasagne.layers.Conv2DLayer(merge9, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)

vis25 = lasagne.layers.get_output(conv9)

conv9 = lasagne.layers.Conv2DLayer(conv9, num_filters= 64, filter_size=(3,3), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(gain='relu'), b=lasagne.init.Constant(0.), nonlinearity=lasagne.nonlinearities.rectify, flip_filters=True, convolution=theano.tensor.nnet.conv2d)


network = lasagne.layers.Conv2DLayer(conv9, num_filters= 1, filter_size=(1,1), stride=(1, 1), pad='same', untie_biases=False, W=lasagne.init.GlorotUniform(), b=lasagne.init.Constant(0.), nonlinearity=None, flip_filters=True, convolution=theano.tensor.nnet.conv2d)


tYhat = lasagne.layers.get_output(network)
tYhat_test = lasagne.layers.get_output(network, deterministic=True)

params = lasagne.layers.get_all_params(network, trainable=True)

filename = "simulated_data_" 

split_path = os.path.splitext(filename)

split_path = os.path.splitext(split_path[0])


try: 

    w_artifact = np.memmap('mice_sparse32_recon.memmap', mode='r', dtype=floatX).reshape(500,128, 128)
    wo_artifact = np.memmap('mice_full_recon.memmap', mode='r', dtype=floatX).reshape(500, 128, 128)

    #print "Reading from memmap files..."

except IOError: # memmap files not found - make them


    w_artifact_file = h5py.File('./dataset/mice_full_recon.mat', 'r')
    variables_w = w_artifact_file.items()

    for var in variables_w:
        name_w_artifact = var[0] 
        data_w_artifact = var[1]
        if type(data_w_artifact) is h5py.Dataset:
            w_artifact = data_w_artifact.value 




    wo_artifact_file = h5py.File('./dataset/mice_sparse32_recon.mat', 'r')

    variables_wo = wo_artifact_file.items()

    for var in variables_wo:    
        name_wo_artifact = var[0] 
        data_wo_artifact = var[1]
        if type(data_wo_artifact) is h5py.Dataset:
            wo_artifact = data_wo_artifact.value # NumPy ndArray / Value

    

num_epochs = 200


saving_path = './saved_results/'


loss = lasagne.objectives.squared_error(tYhat, tY).mean()

learning_rate=0.0005 # fast and smooth

train_fn = get_train_fn(learning_rate)

test_fn = theano.function([tX, tY], [tYhat_test, loss], on_unused_input='ignore')

batchsize = 1

test_fraction = .3


idc_train, idc_test = split_train_test(range(w_artifact.shape[0]), test_fraction)


################################################# load saved model
model_file_first = glob.glob(saving_path +'./model_epoch*')



if  model_file_first != []:
    
    #print "loading from saved model..."    
    with np.load(model_file_first[0]) as n:
        param_values = [n['arr_%d' % j] for j in range(len(n.files))]
    lasagne.layers.set_all_param_values(network, param_values)

    epoch_n = str(model_file_first[0].split('model_epoch')[1])
    e = int(re.search(r'\d+', epoch_n).group())

else:

    e = 1
###################################################


L2_ave_test = []
MSE_ave_test = []
CNR_ave_test = []
SNR_ave_test = []
PSNR_ave_test = []
SSIM_ave_test = []
L1_ave_test = []
dice_ave_test = []



L2_ave_train = []
MSE_ave_train = []
CNR_ave_train = []
SNR_ave_train = []
PSNR_ave_train = []
SSIM_ave_train = []
L1_ave_train = []
dice_ave_train = []

SSIM_ave_input = []

PSNR_ave_input = []
PSNR_ave_inter = []

L1_ave_input = []
L1_ave_inter = []

L2_ave_input = []
L2_ave_inter = []

rel_e_ave_test = []
rel_e_ave_input = []


while e < num_epochs:    

    dice_coeff = []
    L2 = []
    MSE = []
    CNR = []
    SNR = []
    PSNR =[]
    SSIM = []
    L1 = []


    rel_e_input = []
    rel_e = []

    

    SNR_train = []
    PSNR_train = []
    SSIM_train = []
    L2_train = []
    MSE_train = []
    CNR_train = []
    L1_train = []
    dice_coeff_train = []


    SSIM_input = []

    PSNR_inter = []
    PSNR_input = []

    L1_inter = []
    L1_input = []

    L2_inter = []
    L2_input = []

    sys.stdout.write("Epoch {:03d}/{:03d}".format(e, num_epochs))
    sys.stdout.flush()


    start_training_t = time.time()

    artifactual_train = []
    GT_train = []
    predicted_train = []
    intp_train = []

    
    for b in xrange(0, len(idc_train), batchsize):

        idcMB = idc_train[b : b+batchsize]

        xx = np.expand_dims(np.require(w_artifact_train[idcMB, :], dtype = floatX), axis = 0)

        yy = np.require(w_artifact_train[idcMB, :] - wo_artifact_train[idcMB, :], dtype = floatX) 

        
        YhattrainMB, Ltrain = train_fn(xx, yy)

        real_yy = np.squeeze(xx, axis = 0) - yy
        real_ytrain = np.squeeze(xx - YhattrainMB, axis = 0)


        SNR_tmp_train = signaltonoise(real_ytrain, axis = None)
        SNR_train.append(SNR_tmp_train)

        PSNR_tmp_train = calc_psnr(real_yy, real_ytrain)
        PSNR_train.append(PSNR_tmp_train)


        SSIM_tmp_train = ssim(np.squeeze(real_yy, axis=0), np.squeeze(real_ytrain, axis = 0))
        SSIM_train.append(SSIM_tmp_train)

        CNR_tmp_train = 20*np.log10((np.absolute(np.mean(real_yy) - np.mean(real_ytrain - real_yy)))/np.std(real_ytrain - real_yy)) 
        CNR_train.append(CNR_tmp_train)


        L2_tmp_train = np.linalg.norm(real_ytrain - real_yy)
        L2_train.append(L2_tmp_train)


        MSE_tmp_train = sklearn.metrics.mean_squared_error(np.squeeze(real_yy, axis = 0), np.squeeze(real_ytrain,axis = 0)) 
        MSE_train.append(MSE_tmp_train)

        L1_tmp_train = abs(real_ytrain - real_yy).mean() 
        L1_train.append(L1_tmp_train)

        dice_tmp_train = dice(real_ytrain, real_yy)
        dice_coeff_train.append(dice_tmp_train)
   




        if (e+1)%20 == 0:
            learning_rate = 0.8*learning_rate
            artifactualaaa_train = np.squeeze(np.squeeze(xx, axis = 0), axis = 0)
            GTaaa_train = np.squeeze(yy, axis = 0)
            predictedaaa_train = np.squeeze(np.squeeze(YhattrainMB, axis = 0), axis = 0)

            artifactual_train.append(artifactualaaa_train)
            GT_train.append(artifactualaaa_train - GTaaa_train)
            predicted_train.append(artifactualaaa_train - predictedaaa_train)


            artifactual = []
            GT = []
            predicted = []
            intp = []


            for b in xrange(0, len(idc_test), batchsize): 


                idcMB_test = idc_test[b : b+batchsize]

                X_test = np.expand_dims(np.require(w_artifact_test[idcMB_test, : ,:], dtype = floatX), axis = 0)

                Y_test = np.require(w_artifact_test[idcMB_test, : ,:] - wo_artifact_test[idcMB_test, :, :], dtype = floatX)


                Yhattest, Ltest = test_fn(X_test , Y_test)


                real_GT = np.squeeze(X_test, axis = 0) - Y_test 
                real_predicted = np.squeeze(X_test - Yhattest, axis = 0) 


                rel_e_tmp = np.abs(real_GT - real_predicted)
                rel_e.append(rel_e_tmp)

                rel_e_tmp_input = np.abs(real_GT - np.squeeze(X_test, axis = 0))
                rel_e_input.append(rel_e_tmp_input)

                MSE_tmp = sklearn.metrics.mean_squared_error(np.squeeze(real_GT, axis = 0), np.squeeze(real_predicted, axis = 0)) 
                MSE.append(MSE_tmp)

                SNR_tmp = signaltonoise(real_predicted, axis = None)
                SNR.append(SNR_tmp)

                PSNR_tmp = calc_psnr(real_GT, real_predicted)
                PSNR.append(PSNR_tmp)

                PSNR_tmp_input = calc_psnr(real_GT, np.squeeze(X_test, axis = 0))
                PSNR_input.append(PSNR_tmp_input)
		
                SSIM_tmp = ssim(np.squeeze(real_GT, axis=0), np.squeeze(real_predicted, axis = 0))
                SSIM.append(SSIM_tmp)

                SSIM_input_tmp = ssim(np.squeeze(real_GT, axis=0), np.squeeze(np.squeeze(X_test, axis = 0), axis = 0))
                SSIM_input.append(SSIM_input_tmp)

                CNR_tmp = 20*np.log10((np.absolute(np.mean(real_GT) - np.mean(real_predicted - real_GT)))/np.std(real_predicted - real_GT))
                CNR.append(CNR_tmp)

                L2_tmp = np.linalg.norm(np.squeeze(real_predicted,axis =0) - np.squeeze(real_GT,axis =0), 2) 
                L2.append(L2_tmp)

                L2_tmp_input =  np.linalg.norm(np.squeeze(np.squeeze(X_test, axis = 0),axis =0) - np.squeeze(real_GT,axis = 0), 2)
                L2_input.append(L2_tmp_input)

                L1_tmp = abs(real_predicted - real_GT).mean() 
                L1.append(L1_tmp)

                L1_tmp_input =  np.linalg.norm(np.squeeze(np.squeeze(X_test, axis = 0),axis =0) - np.squeeze(real_GT, axis=0), 1)
                L1_input.append(L1_tmp_input)

                dice_tmp = dice(real_predicted, real_GT)
                dice_coeff.append(dice_tmp)



                artifactualaaa = np.squeeze(np.squeeze(X_test, axis = 0), axis = 0)
                GTaaa = np.squeeze(Y_test, axis = 0)
                predictedaaa = np.squeeze(np.squeeze(Yhattest, axis = 0), axis = 0)

                artifactual.append(artifactualaaa)
                GT.append(artifactualaaa - GTaaa)
                predicted.append(artifactualaaa - predictedaaa)

		#sio.savemat(saving_path + 'epoch_' + str(e+1) + '/test/' + '/intp.mat', {"intp" : intp})
                sio.savemat(saving_path + 'epoch_' + str(e+1) + '/test/' + '/artifactual.mat', {"artifactual" : artifactual})
                sio.savemat(saving_path + 'epoch_' + str(e+1) + '/test/' + '/GT.mat', {"GT" : GT})
                sio.savemat(saving_path + 'epoch_' + str(e+1) + '/test/' + '/predicted.mat', {"predicted" : predicted})


		#sio.savemat(saving_path + 'epoch_' + str(e+1) + '/train/' + '/intp_train.mat', {"intp_train" : intp_train})
                sio.savemat(saving_path + 'epoch_' + str(e+1) + '/train/' + '/artifactual_train.mat', {"artifactual_train" : artifactual_train})
                sio.savemat(saving_path + 'epoch_' + str(e+1) + '/train/' + '/GT_train.mat', {"GT_train" : GT_train})
                sio.savemat(saving_path + 'epoch_' + str(e+1) + '/train/' + '/predicted_train.mat', {"predicted_train" : predicted_train})


		##################################### save model
                #print "saving the model..."       
                np.savez_compressed(os.path.join(saving_path +"./model_epoch{:g}.npz".format(e)), *lasagne.layers.get_all_param_values(network))      


		# keep 2 last models!

                model_file = glob.glob('./model_epoch*')
		
                if len(model_file) > 1:

                    a = str(model_file[0].split('model_epoch')[1])
                    b = str(model_file[1].split('model_epoch')[1])
		
                    aa = int(re.search(r'\d+', a).group())
                    bb = int(re.search(r'\d+', b).group())

                    if aa < bb: 
                        os.remove("./model_epoch" + str(aa) + '.npz')
                    else:       
                        os.remove("./model_epoch" + str(bb) + '.npz')
		########################################################


    SNR_ave_test.append(np.average(SNR))
    PSNR_ave_test.append(np.average(PSNR))
    SSIM_ave_test.append(np.average(SSIM))
    L2_ave_test.append(np.average(L2))
    MSE_ave_test.append(np.average(MSE))
    CNR_ave_test.append(np.average(CNR))
    L1_ave_test.append(np.average(L1))
    dice_ave_test.append(np.average(dice_coeff))

    SNR_ave_train.append(np.average(SNR_train))
    PSNR_ave_train.append(np.average(PSNR_train))
    SSIM_ave_train.append(np.average(SSIM_train))
    L2_ave_train.append(np.average(L2_train))
    MSE_ave_train.append(np.average(MSE_train))
    CNR_ave_train.append(np.average(CNR_train))
    L1_ave_train.append(np.average(L1_train))
    dice_ave_train.append(np.average(dice_coeff_train))


    SSIM_ave_input.append(np.average(SSIM_input))
    PSNR_ave_input.append(np.average(PSNR_input))
    PSNR_ave_inter.append(np.average(PSNR_inter))
    L1_ave_inter.append(np.average(L1_inter))
    L1_ave_input.append(np.average(L1_input))
    L2_ave_inter.append(np.average(L2_inter))
    L2_ave_input.append(np.average(L2_input))
    rel_e_ave_test.append(np.average(rel_e))
    rel_e_ave_input.append(np.average(rel_e_input))

    #print "testing took {:g} seconds".format(time.time() - start_testing_t)


    e = e+1


plt.plot(L1_ave_test)
plt.xlabel('epoch')
plt.ylabel('L1_test')
plt.savefig(saving_path + 'L1_test')
plt.close()


plt.plot(dice_ave_test)
plt.xlabel('epoch')
plt.ylabel('dice_test')
plt.savefig(saving_path + 'dice_test')
plt.close()

plt.plot(CNR_ave_test)
plt.xlabel('epoch')
plt.ylabel('20*log CNR_test')
plt.savefig(saving_path + 'CNR_test')
plt.close()

plt.plot(MSE_ave_test)
plt.xlabel('epoch')
plt.ylabel('MSE_test')
plt.savefig(saving_path + 'MSE_test')
plt.close()

plt.plot(SNR_ave_test)
plt.xlabel('epoch')
plt.ylabel('SNR_test')
plt.savefig(saving_path + 'SNR_test')
plt.close()

plt.plot(PSNR_ave_test)
plt.xlabel('epoch')
plt.ylabel('PSNR_test')
plt.savefig(saving_path + 'PSNR_test')
plt.close()

plt.plot(SSIM_ave_test)
plt.xlabel('epoch')
plt.ylabel('SSIM_test')
plt.savefig(saving_path + 'SSIM_test')
plt.close()

plt.plot(L2_ave_test)
plt.xlabel('epoch')
plt.ylabel('L2_test')
#plt.tight_layout()
plt.savefig(saving_path + 'L2_test')
#plt.show()
plt.close()

plt.plot(L1_ave_train)
plt.xlabel('epoch')
plt.ylabel('L1_train')
plt.savefig(saving_path + 'L1_train')
plt.close()

plt.plot(dice_ave_train)
plt.xlabel('epoch')
plt.ylabel('dice_train')
plt.savefig(saving_path + 'dice_train')
plt.close()

plt.plot(L2_ave_train)
plt.xlabel('epoch')
plt.ylabel('L2_train')
plt.savefig(saving_path + 'L2_train')
plt.close()

plt.plot(SNR_ave_train)
plt.xlabel('epoch')
plt.ylabel('SNR_train')
plt.savefig(saving_path + 'SNR_train')
plt.close()

plt.plot(PSNR_ave_train)
plt.xlabel('epoch')
plt.ylabel('PSNR_train')
plt.savefig(saving_path + 'PSNR_train')
plt.close()

plt.plot(SSIM_ave_train)
plt.xlabel('epoch')
plt.ylabel('SSIM_train')
plt.savefig(saving_path + 'SSIM_train')
plt.close()

plt.plot(MSE_ave_train)
plt.xlabel('epoch')
plt.ylabel('MSE_train')
plt.savefig(saving_path + 'MSE_train')
plt.close()

plt.plot(CNR_ave_train)
plt.xlabel('epoch')
plt.ylabel('CNR_train')
plt.savefig(saving_path + 'CNR_train')
plt.close()


np.asarray(rel_e_ave_test)
np.save(saving_path + 'rel_e_ave_test.npy', rel_e_ave_test)

np.asarray(rel_e_ave_input)
np.save(saving_path + 'rel_e_ave_input.npy', rel_e_ave_input)

np.asarray(L2_ave_test)
np.save(saving_path + 'L2_test.npy', L2_ave_test)

np.asarray(MSE_ave_test)
np.save(saving_path + 'MSE_test.npy', MSE_ave_test)

np.asarray(SNR_ave_test)
np.save(saving_path + 'SNR_test.npy', SNR_ave_test)

np.asarray(PSNR_ave_test)
np.save(saving_path + 'PSNR_test.npy', PSNR_ave_test)

np.asarray(SSIM_ave_test)
np.save(saving_path + 'SSIM_test.npy', SSIM_ave_test)

np.asarray(L1_ave_test)
np.save(saving_path + 'L1_test.npy', L1_ave_test)

np.asarray(dice_ave_test)
np.save(saving_path + 'dice_test.npy', dice_ave_test)

np.asarray(CNR_ave_test)
np.save(saving_path + 'CNR_test.npy', CNR_ave_test)

np.asarray(dice_ave_train)
np.save(saving_path + 'dice_train.npy', dice_ave_train)

np.asarray(L1_ave_train)
np.save(saving_path + 'L1_train.npy', L1_ave_train)

np.asarray(L2_ave_train)
np.save(saving_path + 'L2_train.npy', L2_ave_train)

np.asarray(MSE_ave_train)
np.save(saving_path + 'MSE_train.npy', MSE_ave_train)

np.asarray(SNR_ave_train)
np.save(saving_path + 'SNR_train.npy', SNR_ave_train)

np.asarray(PSNR_ave_train)
np.save(saving_path + 'PSNR_train.npy', PSNR_ave_train)

np.asarray(SSIM_ave_train)
np.save(saving_path + 'SSIM_train.npy', SSIM_ave_train)

np.asarray(CNR_ave_train)
np.save(saving_path + 'CNR_train.npy', CNR_ave_train)


