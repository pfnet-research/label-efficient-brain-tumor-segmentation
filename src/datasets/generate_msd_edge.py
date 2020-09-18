import sys
sys.path.append('.')  # NOQA
import copy
import numpy as np
import nibabel as nib
import os
import chainer
from chainer.backends.cuda import get_device_from_id
import chainer.links as L
import cupy

from src.utils.encode_one_hot_vector import encode_one_hot_vector
from src.utils.setup_helpers import _setup_communicator

chainer.backends.cuda.cudnn_enabled = True
chainer.config.use_cudnn = 'always'
chainer.cudnn_deterministic = True

# generates edge information for structseg2019
# dataset using 3D convolution with a Laplacian kernel
dataset = 'nested_brats'
nested = True

if dataset == 'nested_brats':
    print('brats with nested labels!')
    label_path = '/PATH_TO_LABELS/labelsTr/'
    edge_path = './brats_nested_edges/'
    num_range = 484
    nb_class = 4
else:
    print('Could not identify dataset')

if not os.path.isdir(edge_path):
    raise FileNotFoundError('No such directory: {}'.format(edge_path))

# kernel for 3d Laplacian
k = np.array([
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]],
    [[0, 1, 0],
     [1, -6, 1],
     [0, 1, 0]],
    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]], dtype=np.float32)

k = k.reshape((1, 1, 3, 3, 3))
k = k.transpose((0, 1, 3, 4, 2))

default_config = {
    'mn': True,
    'gpu_start_id': 0,
}

config = copy.copy(default_config)
comm, is_master, device = _setup_communicator(config, gpu_start_id=0)

get_device_from_id(device).use()
conv3d = L.Convolution3D(in_channels=1, out_channels=1,
                         ksize=3, stride=1, pad=1, initialW=k)
conv3d.to_gpu()

with chainer.no_backprop_mode(), chainer.using_config('train', False):
    for num in range(1, num_range+1):
        if dataset == 'nested_brats':
            num = "{0:0=3d}".format(num)
            path = label_path+'BRATS_'+str(num)+'.nii.gz'
        try:
            nii_img = nib.load(path)
            affine = nii_img.affine
            img = nii_img.get_data()
            shape = img.shape

            edge = np.zeros((1, nb_class) + shape)
            img = img.reshape((1,)+shape)  # (h,w,d) --> (batch,h,w,d)
            img = encode_one_hot_vector(img, nb_class=nb_class)  # (batch,channel,h,w,d)
            img = cupy.asarray(img, dtype=np.float32)
            for c in range(nb_class):
                if nested:
                    if c == 1:
                        wt = (img[0, 1, :, :, :]).astype(bool)+(img[0, 2, :, :, :]).astype(bool)\
                            + (img[0, 3, :, :, :]).astype(bool)
                        crop_img = wt.reshape((1, 1)+shape)
                    elif c == 2:
                        tc = (img[0, 2, :, :, :]).astype(bool)+(img[0, 3, :, :, :]).astype(bool)
                        crop_img = tc.reshape((1, 1)+shape)
                    elif c == 3:
                        et = (img[0, 3, :, :, :]).astype(bool)
                        crop_img = et.reshape((1, 1)+shape)
                    else:
                        crop_img = img[0, c, :, :, :].reshape((1, 1)+shape)
                else:
                    crop_img = img[0, c, :, :, :].reshape((1, 1)+shape)
                temp_img = conv3d(crop_img.astype(np.float32)).data[0, 0, :, :, :]
                edge[0, c, :, :, :] = (-(cupy.asnumpy(temp_img)) > 0.).astype(int)

            edge_labels_nii = nib.Nifti1Image(edge, affine)
            nib.save(edge_labels_nii, edge_path+'edge_'+str(num)+'.nii.gz')
            print(num)
        except:
            print('No data for '+str(num)+'.nii.gz')
