import os
import random
import numpy as np
import nibabel as nib
import imageio
from skimage.transform import resize
from sklearn.model_selection import KFold, train_test_split


def read_image(path, file_format='nii.gz'):
    """ read image array from path
    Args:
        path (str)          : path to directory which images are stored.
        file_format (str)   : type of reading file {'npy','npz','jpg','png','nii'(3d)}
    Returns:
        image (np.ndarray)  : image array
    """
    path = path + '.' + file_format
    if file_format == 'npy':
        image = np.load(path)
    elif file_format == 'npz':
        image = np.load(path)['arr_0']
    elif file_format in ('png', 'jpg'):
        image = np.array(imageio.imread(path))
    elif file_format == 'dcm':
        image = np.array(imageio.volread(path, 'DICOM'))
    elif file_format in ('nii', 'nii.gz'):
        image = nib.load(path).get_data()
    else:
        raise ValueError('invalid --input_type : {}'.format(file_format))

    return image


def random_crop_pair(image, label, crop_size=(160, 192, 128)):
    H, W, D = crop_size

    if image.shape[1] >= H:
        top = random.randint(0, image.shape[1] - H)
    else:
        raise ValueError('shape of image needs to be larger than output shape')
    h_slice = slice(top, top + H)

    if image.shape[2] >= W:
        left = random.randint(0, image.shape[2] - W)
    else:
        raise ValueError('shape of image needs to be larger than output shape')
    w_slice = slice(left, left + W)

    if image.shape[3] >= D:
        rear = random.randint(0, image.shape[3] - D)
    else:
        raise ValueError('shape of image needs to be larger than output shape')
    d_slice = slice(rear, rear + D)

    image = image[:, h_slice, w_slice, d_slice]
    if label.ndim == 4:
        label = label[:, h_slice, w_slice, d_slice]
    else:
        label = label[h_slice, w_slice, d_slice]
    return image, label


def compute_stats(root_path, arr_type='nii.gz', modality='mri'):
    files = os.listdir(root_path)
    means = []
    stds = []
    for file_i in files:
        img = read_image(os.path.join(root_path, file_i), arr_type)
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=-1)
        img = img.transpose((3, 0, 1, 2))
        if modality == 'ct':
            np.clip(img, -1000., 1000., out=img)
            img += 1000.
        mean = []
        std = []
        for i in range(len(img)):
            mean.append(np.mean(img[i][img[i].nonzero()]))
            if modality == 'ct':
                mean[i] -= 1000.
            std.append(np.std(img[i][img[i].nonzero()]))
        means.append(mean)
        stds.append(std)
    return np.mean(np.array(means), axis=0), np.mean(np.array(stds), axis=0)


def normalize(root_path, arr_type='nii.gz', modality='mri'):
    files = os.listdir(root_path)
    if arr_type == 'nii.gz':
        files = ['.'.join(file_i.split('.')[:-2]) for file_i in files]
    else:
        files = [os.path.splitext(file_i)[0] for file_i in files]
    base_name = os.path.basename(root_path)
    normalized_path = root_path.replace(base_name, base_name+"_normalized")
    if not os.path.exists(normalized_path):
        os.makedirs(normalized_path, exist_ok=True)
    for file_i in files:
        img = read_image(os.path.join(root_path, file_i), arr_type)
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=-1)
        img = img.transpose((3, 0, 1, 2)).astype(np.float32)
        if modality == 'ct':
            np.clip(img, -1000., 1000., out=img)
            img += 1000.
        for i in range(len(img)):
            mean = np.mean(img[i][img[i].nonzero()])
            std = np.std(img[i][img[i].nonzero()])
            img[i] = (img[i] - mean) / std
        img = img.transpose((1, 2, 3, 0))
        np.savez_compressed(os.path.join(normalized_path, file_i.replace(arr_type, 'npz')), img)


def resample_isotropic_nifti(root_path, arr_type='nii.gz', is_label=False):
    files = os.listdir(root_path)
    if arr_type == 'nii.gz':
        files = ['.'.join(file_i.split('.')[:-2]) for file_i in files]
    else:
        files = [os.path.splitext(file_i)[0] for file_i in files]
    base_name = os.path.basename(root_path)
    resampled_path = root_path.replace(base_name, base_name+"_resampled")
    order = 0 if is_label else 1
    if not os.path.exists(resampled_path):
        os.makedirs(resampled_path, exist_ok=True)
    for file_i in files:
        img = read_image(os.path.join(root_path, file_i), 'nii.gz')
        spacing = nib.load(os.path.join(root_path, file_i)).header['pixdim'][:4]
        img = img.transpose((3, 0, 1, 2)).astype(np.float32)
        img_shape = img.shape
        target_shape = tuple([int(img_shape[i] / spacing[i]) for i in range(1, 4)])
        resampled_img = []
        for i in range(len(img)):
            resampled_img.append(resize(img[i], target_shape, order, mode='constant'))
        resampled_img = np.array(resampled_img).transpose((1, 2, 3, 0))
        np.savez_compressed(
            os.path.join(resampled_path, file_i.replace('nii.gz', 'npz')),
            resampled_img)


def split_kfold(root_path, arr_type='nii.gz', n_splits=5, random_state=42):
    data_list = os.listdir(root_path)
    if arr_type == 'nii.gz':
        data_list = ['.'.join(data.split('.')[:-2]) for data in data_list]
    else:
        data_list = [os.path.splitext(data)[0] for data in data_list]
    save_path = root_path.replace(os.path.basename(root_path), 'split_list')
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    # k-fold
    kf = KFold(
        n_splits=n_splits, shuffle=True, random_state=random_state)
    for i, (train_index, val_test_index) in enumerate(kf.split(data_list)):
        val_index, test_index = \
            train_test_split(val_test_index, test_size=0.5, shuffle=True, random_state=random_state)
        val_index = sorted(val_index)
        test_index = sorted(test_index)
        f = open(os.path.join(save_path, 'train_list_cv{}.txt'.format(i)), "w")
        for j in train_index:
            f.write(str(data_list[j]) + "\n")
        f.close()
        f = open(os.path.join(save_path, 'validation_list_cv{}.txt'.format(i)), "w")
        for j in val_index:
            f.write(str(data_list[j]) + "\n")
        f.close()
        f = open(os.path.join(save_path, 'test_list_cv{}.txt'.format(i)), "w")
        for j in test_index:
            f.write(str(data_list[j]) + "\n")
        f.close()
