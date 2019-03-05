import os
import struct
from urllib.request import urlretrieve

import numpy as np

from utils.file_utils import extract_gzip_file

BASE_URL = 'http://yann.lecun.com/exdb/mnist'


def get_mnist_train_images(save_dir: str):
    images = read_idx('train-images-idx3-ubyte', save_dir)
    labels = read_idx('train-labels-idx1-ubyte', save_dir)
    num_images = images.shape[0]
    im_width = images.shape[1]
    im_height = images.shape[2]
    images = _center_images(images).reshape(num_images, im_width * im_height)
    return images, labels


def get_mnist_test_images(save_dir: str):
    images = read_idx('t10k-images-idx3-ubyte', save_dir)
    labels = read_idx('t10k-labels-idx1-ubyte', save_dir)
    num_images = images.shape[0]
    im_width = images.shape[1]
    im_height = images.shape[2]
    images = _center_images(images).reshape(num_images, im_width * im_height)
    return images, labels


def _center_images(images: np.ndarray):
    images = images / 255
    images -= np.mean(images, axis=0)
    return images


def read_idx(filename: str, save_dir: str) -> np.ndarray:
    if filename not in os.listdir(save_dir):
        urlretrieve(os.path.join(BASE_URL, filename) + '.gz', os.path.join(save_dir, filename) + '.gz')
        extract_gzip_file(save_dir, filename + '.gz', filename)

    with open(os.path.join(save_dir, filename), 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
