import numpy as np

from cross_validation import cross_validation_set


def test_cross_validation_set():
    data = np.array([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4], [5, 5]])
    labels = np.array([10, 11, 12, 13, 14, 15])
    count = 0
    for train_data, train_labels, test_data, test_labels in cross_validation_set(data, labels, 3):
        assert len(train_data) == 4
        assert len(train_labels) == 4
        assert len(test_data) == 2
        assert len(test_labels) == 2
        count += 1
    assert count == 3

    count = 0
    for train_data, train_labels, test_data, test_labels in cross_validation_set(data, labels, 2):
        assert len(train_data) == 3
        assert len(train_labels) == 3
        assert len(test_data) == 3
        assert len(test_labels) == 3
        count += 1
    assert count == 2

    train_data_sets = []
    train_label_sets = []
    val_data_sets = []
    val_label_sets = []

    for train_data, train_labels, val_data, val_labels in cross_validation_set(data, labels, 4):
        train_data_sets.append(train_data)
        train_label_sets.append(train_labels)
        val_data_sets.append(val_data)
        val_label_sets.append(val_labels)

    assert len(train_data_sets) == 4
    assert len(train_label_sets) == 4
    assert len(val_data_sets) == 4
    assert len(val_label_sets) == 4

    assert len(train_data_sets[0]) == 4
    assert len(train_label_sets[0]) == 4
    assert len(val_data_sets[0]) == 2
    assert np.array_equal(val_data_sets[0], [[0, 0], [1, 1]])
    assert len(val_label_sets[0]) == 2
    assert np.array_equal(val_label_sets[0], [10, 11])

    assert len(train_data_sets[2]) == 5
    assert len(train_label_sets[2]) == 5
    assert len(val_data_sets[2]) == 1
    assert np.array_equal(val_data_sets[2], [[4, 4]])
    assert len(val_label_sets[2]) == 1
    assert np.array_equal(val_label_sets[2], [14])
