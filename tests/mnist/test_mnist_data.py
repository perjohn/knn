import tempfile

from mnist.mnist_data import get_mnist_train_images, get_mnist_test_images


def test_get_mnist_train_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        images, labels = get_mnist_train_images(temp_dir)
        assert images.shape == (60000, 784)
        assert labels.shape == (60000,)


def test_get_mnist_test_data():
    with tempfile.TemporaryDirectory() as temp_dir:
        images, labels = get_mnist_test_images(temp_dir)
        assert images.shape == (10000, 784)
        assert labels.shape == (10000,)
