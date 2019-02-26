import numpy as np


class KNN:
    def __init__(self, train_images: np.ndarray, train_labels: np.ndarray):
        self.train_images = train_images
        self.train_labels = train_labels

    def predict(self, images: np.ndarray, k_nn: int):
        dists = self._compute_distances(images)
        y_pred = self._predict_labels(dists, k_nn)
        return y_pred

    def _predict_labels(self, dists: np.ndarray, k_nn: int):
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            sorted_idx = np.argsort(dists[i])[:k_nn]
            closest_y = self.train_labels[sorted_idx]
            unique, counts = np.unique(closest_y, return_counts=True)
            y_pred[i] = unique[np.argsort(counts)[-1]]

        return y_pred

    def _compute_distances(self, test_images: np.ndarray) -> np.ndarray:
        x2 = np.sum(self.train_images * self.train_images, axis=1)
        y2 = np.sum(test_images * test_images, axis=1)[None].T
        xy = np.dot(test_images, self.train_images.T)
        return np.sqrt(x2 - 2 * xy + y2)
