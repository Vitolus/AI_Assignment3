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
                sc = SpectralClustering(n_clusters=n_clusters, n_jobs=-1, random_state=1, verbose=2)
                sc.fit(X_train_pca)
                y_pred = _predict(sc.labels_, X_train_pca, pca.transform(self.X_test))
                # Calculate the Rand index
                y_test = cp.asarray(self.y_test)
                n = len(y_test)
                # Calculate pairwise equality for y_test and y_pred
                eq_test = pdist(y_test[:, None], metric='euclidean') == 0
                eq_pred = pdist(y_pred[:, None], metric='euclidean') == 0
                r = 2 * (cp.sum(eq_test & eq_pred) + cp.sum(~eq_test & ~eq_pred)) / (n * (n - 1))
                # Save the best parameters
                if r > best_params[-1]['rand_index']:
                    best_params[-1]['clusters'] = n_clusters
                    best_params[-1]['rand_index'] = r
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')

