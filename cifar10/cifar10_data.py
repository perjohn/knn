import os
import pickle
import tarfile
from urllib.request import urlretrieve

import numpy as np


def get_cifar10_train_images(num_images: int, save_dir: str = None) -> (np.ndarray, np.ndarray):
    batches_dir = extract_data(save_dir)

    images, labels = load_cifar10_train_images(batches_dir)
    return sub_sample(images, labels, num_images)


def get_cifar10_test_images(num_images: int, save_dir: str = None) -> (np.ndarray, np.ndarray):
    batches_dir = extract_data(save_dir)

    images, labels = load_cifar10_test_images(batches_dir)
    return sub_sample(images, labels, num_images)


def load_cifar10_train_images(batches_dir: str) -> (np.ndarray, np.ndarray):
    images = []
    labels = []
    for b in range(1, 6):
        f = os.path.join(batches_dir, 'data_batch_%d' % (b,))
        batch_images, batch_labels = load_cifar10_batch(f)
        images.append(batch_images)
        labels.append(batch_labels)

    images = np.concatenate(images)
    labels = np.concatenate(labels)
    images = np.reshape(images, (images.shape[0], -1))
    return images, labels


def load_cifar10_test_images(batches_dir: str) -> (np.ndarray, np.ndarray):
    f = os.path.join(batches_dir, 'test_batch')
    images, labels = load_cifar10_batch(f)
    images = np.reshape(images, (images.shape[0], -1))
    return images, labels


def load_cifar10_batch(batch_filename):
    with open(batch_filename, 'rb') as f:
        datadict = pickle.load(f, encoding='latin1')
        images = datadict['data']
        labels = datadict['labels']
        images = images.reshape(10000, 3, 32, 32).transpose(0, 2, 3, 1).astype("float")
        labels = np.array(labels)
        return images, labels


def extract_data(save_dir):
    url = 'https://www.cs.toronto.edu/~kriz/'
    tar_filename = 'cifar-10-python.tar.gz'
    extract_dir_name = 'cifar-10-batches-py'
    save_dir = get_or_create_save_dir(save_dir)
    download_tar(save_dir, tar_filename, url)
    batches_dir = os.path.join(save_dir, extract_dir_name)
    extract_tar(batches_dir, save_dir, tar_filename)
    return batches_dir


def extract_tar(batches_dir, save_dir, tar_filename):
    if not os.path.exists(batches_dir):
        extract_tar_file(save_dir, tar_filename)


def download_tar(save_dir, tar_filename, url):
    if tar_filename not in os.listdir(save_dir):
        urlretrieve(os.path.join(url, tar_filename), os.path.join(save_dir, tar_filename))


def get_or_create_save_dir(save_dir):
    if save_dir is None:
        save_dir = os.path.join(os.path.expanduser('~'), 'data', 'CIFAR10')
    os.makedirs(save_dir, exist_ok=True)
    return save_dir


def extract_tar_file(save_dir, tar_filename):
    tar = tarfile.open(os.path.join(save_dir, tar_filename), 'r:gz')
    tar.extractall(path=save_dir)
    tar.close()


def sub_sample(images, labels, num_images):
    mask = list(range(num_images))
    images = images[mask]
    labels = labels[mask]
    return images, labels
