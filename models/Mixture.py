from models.Classifier import Classifier
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class Mixture(Classifier):
    def __init__(self):
        super().__init__()

    def train(self):
        best_params = []
        # Iterate over different PCA dimensions
        for n_components in range(2, 203, 10):
            pca = PCA(n_components=n_components)
            best_params.append({'components': n_components, 'clusters': 0,
                                'time': {'fit': 0.0, 'predict': 0.0}, 'rand_index': 0.0})
            # Iterate over different number of clusters
            for n_clusters in range(5, 16):
                gm = GaussianMixture(n_components=n_clusters, covariance_type='diag',
                                     random_state=1, verbose=1)
                print(f'Training PCA with {n_components} components and {n_clusters} clusters')
                self._fit_predict(gm, pca, n_clusters, best_params)
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')
