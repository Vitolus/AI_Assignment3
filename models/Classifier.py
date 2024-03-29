import time

from sklearn.datasets import fetch_openml
from atom import ATOMClassifier
import cupy as cp
from cupyx.scipy.spatial.distance import pdist


class Classifier:
    def __init__(self, n_rows=70000, test_size=10000):
        self.X, self.y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        self.y = self.y.astype(int)  # convert string to int
        self.X = self.X / 255.  # normalize data
        self.model = ATOMClassifier(self.X, self.y, test_size=test_size, n_jobs=-1, n_rows=n_rows,
                                    device="cpu", engine="sklearn", verbose=2, random_state=1)
        self.X_train = self.model.X_train.to_numpy()
        self.X_test = self.model.X_test.to_numpy()
        self.y_train = self.model.y_train.to_numpy()
        self.y_test = self.model.y_test.to_numpy()

    def _fit_predict(self, model, pca, k, best_params):
        start = time.perf_counter()
        model.fit(pca.fit_transform(self.X_train))
        end_fit = time.perf_counter() - start
        start = time.perf_counter()
        y_pred = cp.asarray(model.predict(pca.transform(self.X_test)))
        end_pre = time.perf_counter() - start
        self._compute_rand_index(y_pred, k, best_params, end_fit, end_pre)

    def _compute_rand_index(self, y_pred, k, best_params, end_fit, end_pre):
        y_test = cp.asarray(self.y_test)
        n = len(y_test)
        # Calculate pairwise equality for y_test and y_pred
        eq_test = pdist(y_test[:, None], metric='euclidean') == 0
        eq_pred = pdist(y_pred[:, None], metric='euclidean') == 0
        r = 2 * (cp.sum(eq_test & eq_pred) + cp.sum(~eq_test & ~eq_pred)) / (n * (n - 1))
        # Save the best parameters
        if r > best_params[-1]['rand_index']:
            best_params[-1]['clusters'] = k
            best_params[-1]['rand_index'] = r
            best_params[-1]['time']['fit'] = end_fit
            best_params[-1]['time']['predict'] = end_pre

