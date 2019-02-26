import numpy as np
import struct


def get_mnist_train_images():
    images = read_idx('/Users/perjohn/data/MNIST/train-images-idx3-ubyte')
    labels = read_idx('/Users/perjohn/data/MNIST/train-labels-idx1-ubyte')
    num_images = images.shape[0]
    im_width = images.shape[1]
    im_height = images.shape[2]
    images = _center_images(images).reshape(num_images, im_width * im_height)
    return images, labels


def get_mnist_test_images():
    images = read_idx('/Users/perjohn/data/MNIST/t10k-images-idx3-ubyte')
    labels = read_idx('/Users/perjohn/data/MNIST/t10k-labels-idx1-ubyte')
    num_images = images.shape[0]
    im_width = images.shape[1]
    im_height = images.shape[2]
    images = _center_images(images).reshape(num_images, im_width * im_height)
    return images, labels


def _center_images(images: np.ndarray):
    images = images / 255
    images -= np.mean(images, axis=0)
    return images


def read_idx(filename: str) -> np.ndarray:
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for _ in range(dims))
        return np.fromstring(f.read(), dtype=np.uint8).reshape(shape)
