from models.Classifier import Classifier
from sklearn.decomposition import PCA
from sklearn.cluster import MeanShift


class Shift(Classifier):
    def __init__(self):
        super().__init__()

    def train(self):
        best_params = []
        # Iterate over different PCA dimensions
        for n_components in range(2, 203, 10):
            pca = PCA(n_components=n_components)
            best_params.append({'components': n_components, 'clusters': 0,
                                'time': {'fit': 0.0, 'predict': 0.0}, 'rand_index': 0.0})
            # Iterate over different kernel widths
            for bandwidth in range(5, 16):
                ms = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=False,
                               n_jobs=-1)
                print(f'Training PCA with {n_components} components and {bandwidth} bandwidth')
                self._fit_predict(ms, pca, bandwidth, best_params)
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')
