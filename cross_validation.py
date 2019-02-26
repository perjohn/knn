import numpy as np


def cross_validation_set(images: np.ndarray, labels: np.ndarray, num_folds: int):
    images_folds = np.array_split(images, num_folds)
    labels_folds = np.array_split(labels, num_folds)

    for i in range(0, num_folds):
        train_images = np.vstack(images_folds[:i] + images_folds[i + 1:])
        train_labels = np.hstack(labels_folds[:i] + labels_folds[i + 1:])
        val_images = images_folds[i]
        val_labels = labels_folds[i]
        yield train_images, train_labels, val_images, val_labels
