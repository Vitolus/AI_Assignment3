from models.Classifier import Classifier
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import cupy as cp
from sklearn.metrics.pairwise import rbf_kernel
import time


def _predict(labels, X_train, X_test, k=10):
    similarity = rbf_kernel(X_test, X_train)
    k_idx = cp.argsort(similarity, axis=1)[:, -k:]
    k_labels = labels[k_idx]
    return cp.asarray([cp.argmax(cp.bincount(cp.asarray(label))) for label in k_labels])


class Cut(Classifier):
    def __init__(self):
        super().__init__(n_rows=10000, test_size=5800)

    def train(self):
        best_params = []
        # Iterate over different PCA dimensions
        for n_components in range(2, 203, 10):
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(self.X_train)
            best_params.append({'components': n_components, 'clusters': 0,
                                'time': {'fit': 0.0, 'predict': 0.0}, 'rand_index': 0.0})
            # Iterate over different number of clusters
            for n_clusters in range(5, 16):
                sc = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr',
                                        n_jobs=-1, random_state=1)
                print(f'Training PCA with {n_components} components and {n_clusters} clusters')
                start = time.perf_counter()
                sc.fit(X_train_pca)
                end_fit = time.perf_counter() - start
                start = time.perf_counter()
                y_pred = _predict(sc.labels_, X_train_pca, pca.transform(self.X_test))
                end_pre = time.perf_counter() - start
                self._compute_rand_index(y_pred, n_clusters, best_params, end_fit, end_pre)
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')

