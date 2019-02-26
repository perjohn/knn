import tempfile

from cifar10_data import get_cifar10_train_images, get_cifar10_test_images


def test_get_cifar10_train_images():
    with tempfile.TemporaryDirectory() as temp_dir:
        images, labels = get_cifar10_train_images(100, temp_dir)
        assert images.shape == (100, 3072)
        assert labels.shape == (100,)


def test_get_cifar10_test_images():
    with tempfile.TemporaryDirectory() as temp_dir:
        images, labels = get_cifar10_test_images(50, temp_dir)
        assert images.shape == (50, 3072)
        assert labels.shape == (50,)
