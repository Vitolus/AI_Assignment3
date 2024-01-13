from sklearn.datasets import fetch_openml
from atom import ATOMClassifier


class Classifier:
    def __init__(self):
        self.X, self.y = fetch_openml('mnist_784', version=1, return_X_y=True)  # load data
        self.y = self.y.astype(int)  # convert string to int
        self.X = self.X / 255.  # normalize data
        self.model = ATOMClassifier(self.X, self.y, test_size=10000, n_jobs=-1,
                                    device="cpu", engine="sklearn", verbose=2, random_state=1)
        self.results = None

    def train(self):
        self.model.feature_selection(strategy="pca", solver="full")
