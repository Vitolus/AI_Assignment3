from models.Classifier import Classifier
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift
import cupy as cp
from cupyx.scipy.spatial.distance import pdist
import time


class Shift(Classifier):
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
                ms = MeanShift(bandwidth=bandwidth,
                               n_jobs=-1)
                self._fit_predict(ms, pca, bandwidth, best_params)
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')
