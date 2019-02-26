import numpy as np

from k_nearest_neighbours.knn import KNN


def test_compute_distances():
    train = np.zeros((3, 4), dtype=float)
    train[1] = np.ones((1, 4), dtype=float)
    train[2] = np.array([2, 2, 2, 2], dtype=float)
    gt = np.zeros((2,))
    gt[1] = 1.

    k_nearest_neighbor = KNN(train, gt)

    test = np.zeros((2, 4), dtype=float)
    test[1] = np.ones((1, 4), dtype=float)

    dists = k_nearest_neighbor._compute_distances(test)
    assert dists.shape == (2, 3)
    assert dists[0, 0] == 0.
    assert dists[0, 1] == 2.
    assert dists[0, 2] == 4.
    assert dists[1, 0] == 2.
    assert dists[1, 1] == 0.
    assert dists[1, 2] == 2.


def test_predict_single_train_image():
    train_images = np.ones((1, 16), dtype=np.int8)
    train_labels = np.ones((1,), dtype=np.int8)

    image = np.ones((1, 16), dtype=np.int8)
    knn = KNN(train_images, train_labels)
    result = knn.predict(image, 1)
    assert result == 1


def test_predict_single_test_image():
    train_image_1 = np.ones((1, 16), dtype=np.int8)
    train_image_2 = np.zeros((1, 16), dtype=np.int8)
    train_images = np.vstack((train_image_1, train_image_2))
    train_labels = np.arange(0, 2)

    image = np.ones((1, 16), dtype=np.int8)
    knn = KNN(train_images, train_labels)
    result = knn.predict(image, 1)
    assert result == 0

    image = np.zeros((1, 16), dtype=np.int8)
    knn = KNN(train_images, train_labels)
    result = knn.predict(image, 1)
    assert result == 1


def test_predict_two_test_images():
    train_image_1 = np.ones((1, 16), dtype=np.int8)
    train_image_2 = np.zeros((1, 16), dtype=np.int8)
    train_images = np.vstack((train_image_1, train_image_2))
    train_labels = np.arange(0, 2)

    test_image_1 = np.ones((1, 16), dtype=np.int8)
    test_image_2 = np.zeros((1, 16), dtype=np.int8)

    test_images = np.vstack([test_image_1, test_image_2])
    knn = KNN(train_images, train_labels)
    result = knn.predict(test_images, 1)
    assert len(result) == 2
    assert result[0] == 0
    assert result[1] == 1
