import click
import numpy as np

from cross_validation.cross_validation import run_cross_validation
from k_nearest_neighbours.knn import KNN
from mnist_data import get_mnist_test_images, get_mnist_train_images


@click.group()
def cli():
    pass


@cli.command()
@click.option('--k-nn', default=1, type=click.INT)
def test(k_nn):
    train_images, train_labels = get_mnist_train_images()

    test_images, test_labels = get_mnist_test_images()
    knn = KNN(train_images, train_labels)

    pred = knn.predict(test_images, k_nn)
    num_correct = np.sum(pred == test_labels)
    accuracy = float(num_correct) / len(test_images)
    print('Accuracy: {}'.format(accuracy))


@cli.command()
def cross_validate():
    images, labels = get_mnist_train_images()
    k_nns = [1, 3, 5, 8, 10, 12, 15, 20, 50, 100]

    run_cross_validation(images, labels, k_nns)


if __name__ == "__main__":
    cli()
