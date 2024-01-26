from models.Classifier import Classifier
from sklearn.decomposition import PCA
from sklearn.cluster import SpectralClustering
import cupy as cp
from cupyx.scipy.spatial.distance import pdist
from sklearn.metrics.pairwise import rbf_kernel


def _predict(labels, X_train, X_test, k=10):
    similarity = rbf_kernel(X_test, X_train)
    k_idx = cp.argsort(similarity, axis=1)[:, -k:]
    k_labels = labels[k_idx]
    return cp.asarray([cp.argmax(cp.bincount(label)) for label in k_labels])


class Mixture(Classifier):
    def __init__(self):
        super().__init__()

    def train(self):
        best_params = []
        # Iterate over different PCA dimensions
        for n_components in range(2, 203, 10):
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(self.X_train)
            best_params.append({'components': n_components, 'clusters': 0, 'rand_index': 0.0})
            # Iterate over different number of clusters
            for n_clusters in range(5, 16):
                sc = SpectralClustering(n_clusters=n_clusters, assign_labels='cluster_qr',
                                        n_jobs=-1, random_state=1, verbose=2)
                sc.fit(X_train_pca)
                y_pred = _predict(sc.labels_, X_train_pca, pca.transform(self.X_test))
                self._compute_rand_index(y_pred, n_clusters, best_params)
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')

