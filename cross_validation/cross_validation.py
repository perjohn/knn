import numpy as np

from k_nearest_neighbours.knn import KNN


def run_cross_validation(images: np.ndarray, labels: np.ndarray, k_nns: list):
    for k_nn in k_nns:
        accuracies = []
        for train_images, train_labels, val_images, val_labels in cross_validation_set(images, labels, 5):
            knn = KNN(train_images, train_labels)
            pred = knn.predict(val_images, k_nn)
            num_correct = np.sum(pred == val_labels)
            accuracy = float(num_correct) / len(val_images)
            print('Accuracy for k={} is {}'.format(k_nn, accuracy))
            accuracies.append(accuracy)
        print('Average accuracy for k={} is {}'.format(k_nn, sum(accuracies) / 5))


def cross_validation_set(images: np.ndarray, labels: np.ndarray, num_folds: int):
    images_folds = np.array_split(images, num_folds)
    labels_folds = np.array_split(labels, num_folds)

    for i in range(0, num_folds):
        train_images = np.vstack(images_folds[:i] + images_folds[i + 1:])
        train_labels = np.hstack(labels_folds[:i] + labels_folds[i + 1:])
        val_images = images_folds[i]
        val_labels = labels_folds[i]
        yield train_images, train_labels, val_images, val_labels
