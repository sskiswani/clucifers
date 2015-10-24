import numpy as np
from scipy import stats
import logging
from typing import Iterable, Union
from .core import Classifier

logger = logging.getLogger(__name__)

def make_discriminant(W, w, omega):
    return lambda x: float(((x.T).dot(W)).dot(x) + (w.T).dot(x) + omega)


class BayesMLE(Classifier):
    def __init__(self, training_data: Union[Iterable[np.ndarray], np.ndarray], **kwargs):
        super().__init__(training_data, **kwargs)

        classes = training_data[:, 0]
        self.parameters = {}
        for c in self.priors.keys():
            features = training_data[np.where(classes == c)][:, 1:]

            mu = np.sum(features, axis=0) / np.sum(classes == c)

            memo = features - mu
            sigma = memo.T.dot(memo) / np.sum(classes == c)
            mu = mu.reshape(mu.shape[0], 1)

            # Calculate discriminant function
            psigma = sigma.copy()
            pmu = mu.copy()
            W = (-1 / 2) * np.linalg.inv(sigma)
            w = np.linalg.inv(sigma).dot(mu)

            omega = ((((-1 / 2) * (mu).T).dot(np.linalg.inv(sigma))).dot(mu)) - (-1 / 2) * np.log(np.linalg.det(sigma))

            self.parameters[c] = {
                'mu':pmu.copy(),
                'sigma':psigma.copy(),
                'g': make_discriminant(W, w, omega)
            }
        # end loop

    def __repr__(self):
        return str(self)


class GaussianMLE(BayesMLE):
    pass
