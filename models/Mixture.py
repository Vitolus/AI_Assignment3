import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


class Mixture:
    def __init__(self):
        self.X, self.y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        self.y = self.y.astype(int)  # convert string to int
        self.X = self.X / 255.  # normalize data

    def fit(self):
        # Iterate over different PCA dimensions
        for n_components in range(2, 201):
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(self.X)

            # Iterate over different number of clusters
            for n_clusters in range(5, 16):
                gm = GaussianMixture(n_components=n_clusters, covariance_type='diag')
                gm.fit(X_pca)
                y_pred = gm.predict(X_pca)

                # Calculate the Rand index
                n = len(self.y)
                a = 0
                b = 0
                for i in range(n):
                    for j in range(i + 1, n):
                        if self.y[i] == self.y[j] and y_pred[i] == y_pred[j]:
                            a += 1
                        elif self.y[i] != self.y[j] and y_pred[i] != y_pred[j]:
                            b += 1
                r = 2 * (a + b) / (n * (n - 1))

                print(f'PCA: {n_components}, Clusters: {n_clusters}, Rand index: {r:.4f}')
