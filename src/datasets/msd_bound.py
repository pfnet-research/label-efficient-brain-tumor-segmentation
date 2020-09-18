import os
import numpy as np
from chainer.dataset import DatasetMixin
from src.datasets.preprocess import read_image, random_crop_pair
from src.functions.array.resize_images_3d import resize_images_3d
from src.functions.array.resize_images_3d_nearest_neighbor import resize_images_3d_nearest_neighbor
from numpy.random import rand
import chainer
import chainer.functions as F


class MSDBoundDataset(DatasetMixin):
    """Edited MSD dataset for 4-channel labels (for boundary aware networks)"""
    def __init__(self, config, split_file):
        self.crop_size = eval(config['crop_size'])
        self.shift_intensity = config['shift_intensity']
        self.random_flip = config['random_flip']
        self.target_label = config['target_label']
        self.nb_class = config['nb_labels']
        self.unet_cropping = config['unet_cropping']
        self.random_scale = config['random_scale']
        self.training = True
        with open(split_file) as f:
            self.split_list = [line.rstrip() for line in f]
        image_list = [os.path.join(config['image_path'], name) for name in self.split_list]
        label_list = [os.path.join(config['label_path'], name) for name in self.split_list]
        edge_list = [os.path.join(config['edge_path'], 'edge_'+str(name)[-3:])
                     for name in self.split_list]
        self.edge_label = config['edge_label']
        assert len(image_list) == len(label_list)
        self.pair_list = list(zip(image_list, label_list, edge_list))
        self.nb_copies = config['structseg_nb_copies']
        self.image_file_format = config['image_file_format']
        self.ignore_path = config['ignore_path']
        self.nested_label = config['nested_label']

    def __len__(self):
        return len(self.pair_list) * self.nb_copies

    def _get_image(self, i):
        i = i % len(self.pair_list)
        image = read_image(self.pair_list[i][0], self.image_file_format)
        image = image.transpose((3, 0, 1, 2))
        return image

    def _get_label(self, i):
        i = i % len(self.pair_list)
        label = read_image(self.pair_list[i][1], 'nii.gz')
        if self.training and (self.ignore_path is not None):
            # use samples without labels and use subtask branch only
            ignore_list = open(self.ignore_path).read().splitlines()
            if self.pair_list[i][1][-9:] not in ignore_list:
                # giving NaN label when the sample is not in the ignore_list
                # ignore_list: list of samples which we DO NOT ignore labels
                label = np.empty(label.shape)
                label[:] = np.nan
        if self.unet_cropping:
            label = label[20:-20, 20:-20, 20:-20]
        return label

    def _get_edge(self, i):
        i = i % len(self.pair_list)
        edge = read_image(self.pair_list[i][2], 'nii.gz')
        edge = edge[0, :, :, :, :]
        if self.unet_cropping:
            edge = edge[20:-20, 20:-20, 20:-20]
        return edge

    def _crop(self, x, y):
        return random_crop_pair(x, y, self.crop_size)

    def _change_intensity(self, x):
        for i in range(len(x)):
            diff = 2 * self.shift_intensity * rand() - self.shift_intensity
            x[i] += diff
        return x

    @staticmethod
    def _flip(x, y):
        if np.random.random() < 0.5:
            x = x[:, ::-1, :, :]
            y = y[:, ::-1, :, :]
        if np.random.random() < 0.5:
            x = x[:, :, ::-1, :]
            y = y[:, :, ::-1, :]
        if np.random.random() < 0.5:
            x = x[:, :, :, ::-1]
            y = y[:, :, :, ::-1]
        return x, y

    @staticmethod
    def _binarize_label(y, t_idx):
        return (y >= t_idx).astype(np.int32)

    @staticmethod
    def _binarize_brats(y, dc=1):
        if (np.sum(y) < 0) or (np.isnan(y).any()):
            # when using ignore list (semi-supervised VAEseg)
            return y
        else:
            if dc == 1:
                whole_tumor = (y >= 1).astype(np.int32)
                return whole_tumor
            elif dc == 2:
                tumor_core = ((y == 3) + (y == 2)).astype(np.int32)
                return tumor_core
            elif dc == 3:
                enhancing_tumor = (y == 3).astype(np.int32)
                return enhancing_tumor
            else:
                print('Error for binarization')
                return y

    @staticmethod
    def _rand_scale(x, y):
        # random scale the image [0.9,1.1) for augmentation
        xx = x.reshape((1,)+x.shape)
        y = y.astype(np.float32)

        rand_scale = 0.9+rand()*0.2
        orig_shape = np.array(xx.shape)
        scaled_shape = orig_shape[2:]*rand_scale
        scaled_shape = np.round(scaled_shape).astype(int)

        resized_x = resize_images_3d(xx, scaled_shape)
        resized_y = resize_images_3d_nearest_neighbor(y.reshape((1,)+y.shape),
                                                      scaled_shape)[0, :, :, :, :]
        resized_y = chainer.Variable(resized_y)
        if rand_scale < 1:
            # zero-padding if the image is scaled down
            pad_w = orig_shape[2] - scaled_shape[0]
            pad_h = orig_shape[3] - scaled_shape[1]
            pad_d = orig_shape[4] - scaled_shape[2]

            w_half = int(round(pad_w/2))
            h_half = int(round(pad_h/2))
            d_half = int(round(pad_d/2))

            pad_width_x = ((0, 0), (0, 0), (w_half, pad_w-w_half),
                           (h_half, pad_h-h_half), (d_half, pad_d-d_half))
            pad_width_y = ((0, 0), (w_half, pad_w-w_half),
                           (h_half, pad_h-h_half), (d_half, pad_d-d_half))
            resized_x = F.pad(resized_x, pad_width=pad_width_x, mode='constant', constant_values=0.)
            resized_y = F.pad(resized_y, pad_width=pad_width_y, mode='constant', constant_values=0.)
        assert resized_x.shape[2:] == resized_y.shape[1:]
        resized_x = resized_x[0, :, :, :, :]

        return resized_x.data, resized_y.data.astype(np.int32)

    def get_example(self, i):
        x = self._get_image(i)
        y = self._get_label(i)
        if self.edge_label:
            y = y.reshape((1,)+y.shape)
            ye = self._get_edge(i)
            y = np.append(y, ye, axis=0)
        else:
            y = y.reshape((1,)+y.shape)

        if self.random_scale:
            x, y = self._rand_scale(x, y)
        if self.training:
            x, y = self._crop(x, y)
        if self.shift_intensity > 0.:
            x = self._change_intensity(x)
        if self.random_flip:
            x, y = self._flip(x, y)
        if self.target_label:
            assert self.nb_class == 2
            y = self._binarize_label(y, self.target_label)
        elif self.nested_label:
            ys = []
            for i in range(self.nb_class-1):
                ys.append(self._binarize_brats(y[0, :, :, :], i+1))
            if self.edge_label:
                return x.astype(np.float32),\
                       np.array(ys).astype(np.int32), y[1:, :, :, :].astype(np.int32)
            else:
                return x.astype(np.float32), np.array(ys).astype(np.int32)

        if self.edge_label:
            return x.astype(np.float32), y[0, :, :, :].astype(np.int32),\
                   y[1:, :, :, :].astype(np.int32)
        else:
            return x.astype(np.float32), y[0, :, :, :].astype(np.int32)
