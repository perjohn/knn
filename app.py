import click
import numpy as np

from cifar10.cifar10_data import get_cifar10_train_images, get_cifar10_test_images
from cross_validation.cross_validation import run_cross_validation
from k_nearest_neighbours.knn import KNN
from mnist.mnist_data import get_mnist_test_images, get_mnist_train_images


@click.group()
@click.pass_context
def cli(ctx):
    pass


_global_options = [
    click.option('--k-nn', default=1, type=click.INT)
]


def global_options(func):
    for option in reversed(_global_options):
        func = option(func)
    return func


@cli.command()
@global_options
def mnist_test(k_nn):
    train_images, train_labels = get_mnist_train_images()
    test_images, test_labels = get_mnist_test_images()

    accuracy = calculate_knn_accuracy(k_nn, test_images, test_labels, train_images, train_labels)
    print('Accuracy: {}'.format(accuracy))


@cli.command()
def mnist_cross_validate():
    images, labels = get_mnist_train_images()
    k_nns = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    run_cross_validation(images, labels, k_nns)


@cli.command()
@global_options
def cifar10_test(k_nn):
    train_images, train_labels = get_cifar10_train_images(10000)
    test_images, test_labels = get_cifar10_test_images(1000)

    accuracy = calculate_knn_accuracy(k_nn, test_images, test_labels, train_images, train_labels)
    print('Accuracy: {}'.format(accuracy))


@cli.command()
def cifar10_cross_validate():
    images, labels = get_cifar10_train_images(5000)
    k_nns = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    run_cross_validation(images, labels, k_nns)


def calculate_knn_accuracy(k_nn, test_images, test_labels, train_images, train_labels):
    knn = KNN(train_images, train_labels)
    pred = knn.predict(test_images, k_nn)
    num_correct = np.sum(pred == test_labels)
    accuracy = float(num_correct) / len(test_images)
    return accuracy


if __name__ == "__main__":
    cli()
