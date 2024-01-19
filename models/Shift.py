from models.Classifier import Classifier
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
import cupy as cp
from cupyx.scipy.spatial.distance import pdist


class Mixture(Classifier):
    def __init__(self):
        super().__init__()

    def train(self):
        best_params = []
        # Iterate over different PCA dimensions
        for n_components in range(2, 203, 10):
            pca = PCA(n_components=n_components)
            best_params.append({'components': n_components, 'kernel_width': 0.0, 'rand_index': 0.0})
            # Iterate over different kernel widths
            for bandwidth in range(5, 16):
                ms = MeanShift(bandwidth=bandwidth, n_jobs=-1)
                ms.fit(pca.fit_transform(self.X_train))
                y_pred = cp.asarray(ms.predict(pca.transform(self.X_test)))
                # Calculate the Rand index
                y_test = cp.asarray(self.y_test)
                n = len(y_test)
                # Calculate pairwise equality for y_test and y_pred
                eq_test = pdist(y_test[:, None], metric='euclidean') == 0
                eq_pred = pdist(y_pred[:, None], metric='euclidean') == 0
                r = 2 * (cp.sum(eq_test & eq_pred) + cp.sum(~eq_test & ~eq_pred)) / (n * (n - 1))
                # Save the best parameters
                if r > best_params[-1]['rand_index']:
                    best_params[-1]['kernel_width'] = bandwidth
                    best_params[-1]['rand_index'] = r
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')
