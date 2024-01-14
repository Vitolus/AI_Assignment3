from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from atom import ATOMClassifier
import matplotlib.pyplot as plt


class Mixture:
    def __init__(self):
        self.X, self.y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        self.y = self.y.astype(int)  # convert string to int
        self.X = self.X / 255.  # normalize data
        self.model = ATOMClassifier(self.X, self.y, test_size=1000, n_jobs=-1, n_rows=5000,
                                    device="clu", engine="sklearn", verbose=2, random_state=1)
        self.X_train = self.model.X_train.to_numpy()
        self.X_test = self.model.X_test.to_numpy()
        self.y_train = self.model.y_train.to_numpy()
        self.y_test = self.model.y_test.to_numpy()

    def train(self):
        best_params = {
            'components': 0,
            'clusters': 0,
            'rand_index': 0
        }
        # Iterate over different PCA dimensions
        for n_components in range(2, 201, 10):
            pca = PCA(n_components=n_components)
            X_train_pca = pca.fit_transform(self.X_train)
            X_test_pca = pca.transform(self.X_test)

            # Iterate over different number of clusters
            for n_clusters in range(5, 16):
                gm = GaussianMixture(n_components=n_clusters, covariance_type='diag')
                gm.fit(X_train_pca)
                y_pred = gm.predict(X_test_pca)
                # Calculate the Rand index
                n = len(self.y_test)
                a = 0
                b = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        print(f'Progress: {i}/{n}, {j}/{n}', end='\r')
                        if self.y_test[i] == self.y_test[j] and y_pred[i] == y_pred[j]:
                            a += 1
                        elif self.y_test[i] != self.y_test[j] and y_pred[i] != y_pred[j]:
                            b += 1
                r = 2 * (a + b) / (n * (n - 1))
                # Update the best parameters
                if r > best_params['rand_index']:
                    best_params['components'] = n_components
                    best_params['clusters'] = n_clusters
                    best_params['rand_index'] = r
                print(f'PCA: {n_components}, Clusters: {n_clusters}, Rand index: {r:.4f}')
