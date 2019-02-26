import click
import numpy as np

from cifar10_data import get_cifar10_test_images, get_cifar10_train_images
from cross_validation import cross_validation_set
from knn import KNN


@click.group()
def cli():
    pass


@cli.command()
@click.option('--k-nn', default=1, type=click.INT)
def test(k_nn):
    train_images, train_labels = get_cifar10_train_images(10000)
    test_images, test_labels = get_cifar10_test_images(1000)
    knn = KNN(train_images, train_labels)

    pred = knn.predict(test_images, k_nn)
    num_correct = np.sum(pred == test_labels)
    accuracy = float(num_correct) / len(test_images)
    print('Accuracy: {}'.format(accuracy))


@cli.command()
def cross_validation():
    images, labels = get_cifar10_train_images(5000)
    k_nns = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

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


if __name__ == "__main__":
    cli()
