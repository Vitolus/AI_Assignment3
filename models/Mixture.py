from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from atom import ATOMClassifier
from itertools import combinations


class Mixture:
    def __init__(self):
        self.X, self.y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        self.y = self.y.astype(int)  # convert string to int
        self.X = self.X / 255.  # normalize data
        self.model = ATOMClassifier(self.X, self.y, test_size=2000, n_jobs=-1, n_rows=10000,
                                    device="clu", engine="sklearn", verbose=2, random_state=1)
        self.X_train = self.model.X_train.to_numpy()
        self.X_test = self.model.X_test.to_numpy()
        self.y_train = self.model.y_train.to_numpy()
        self.y_test = self.model.y_test.to_numpy()

    def train(self):
        best_params = []
        # Iterate over different PCA dimensions
        for n_components in range(2, 203, 10):
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(self.X_train)
            X_test_pca = pca.transform(self.X_test)
            best_params.append({'components': n_components, 'clusters': 0, 'rand_index': 0.0})
            # Iterate over different number of clusters
            for n_clusters in range(5, 16):
                gm = GaussianMixture(n_components=n_clusters, covariance_type='diag')
                gm.fit(X_train_pca)
                y_pred = gm.predict(X_test_pca)
                # Calculate the Rand index
                n = len(self.y_test)
                a = 0
                b = 0
                for i, j in combinations(range(n), 2):
                    if self.y_test[i] == self.y_test[j] and y_pred[i] == y_pred[j]:
                        a += 1
                    elif self.y_test[i] != self.y_test[j] and y_pred[i] != y_pred[j]:
                        b += 1
                    print(f'Progress: Components: {n_components}, Clusters: {n_clusters}, {i}/{n}, {j}/{n}', end='\r')
                r = 2 * (a + b) / (n * (n - 1))
                # Save the best parameters
                if r > best_params[-1]['rand_index']:
                    best_params[-1]['clusters'] = n_clusters
                    best_params[-1]['rand_index'] = r
            # Print the best parameters
            print(f'Best parameters for {n_components} components: {best_params[-1]}')
