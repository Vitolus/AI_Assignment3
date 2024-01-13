from atom import ATOMModel
from sklearn.mixture import GaussianMixture
import Classifier


class Mixture(Classifier):
    def __init__(self):
        super().__init__()
        self.mixture = ATOMModel(estimator=GaussianMixture,acronym="GM", name="Gaussian Mixture", needs_scaling=True)

    def train(self):
        super().train()
        self.model.run(
            models="",
            metric="",
            n_trials=10,
            parallel=True,
            est_params={
                "covariance_type": "diag",
            },
            ht_params={
                "distributions": {

                },
            }
        )
        self.results = self.model.evaluate()
