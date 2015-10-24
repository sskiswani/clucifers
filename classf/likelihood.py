import numpy as np
import logging
from typing import Iterable, Union
from .core import Classifier

logger = logging.getLogger(__name__)

def make_discriminant(mu, sigma, prior):
    mat_mu = np.matrix(mu)
    invSigma = np.linalg.inv(sigma)

    Wi = np.matrix(-0.5 * invSigma)
    wi = np.matrix(invSigma * mat_mu)
    wi0 = -0.5*(mat_mu.T).dot(np.linalg.inv(sigma)).dot(mat_mu) - 0.5*np.log(np.linalg.det(sigma)) + np.log(prior)

    return lambda x: np.float(-0.5*np.matrix(x).dot(Wi).dot(np.matrix(x).T) + (wi.T).dot(x) + wi0)

class BayesMLE(Classifier):
    def __init__(self, training_data: Union[Iterable[np.ndarray], np.ndarray], **kwargs):
        super().__init__(training_data, **kwargs)

        classes = training_data[:, 0]
        self.parameters = {}
        for c in self.priors.keys():
            features = training_data[np.where(classes == c)][:, 1:]

            mu = np.sum(features, axis=0) / np.sum(classes == c)

            memo = np.matrix(features - mu)
            sigma = memo.T.dot(memo) / np.sum(classes == c)
            mu = mu.reshape(mu.shape[0], 1)

            self.parameters[c] = {
                'mu':mu,
                'sigma':sigma,
                'g': make_discriminant(mu.copy(), sigma.copy(), self.priors[c])
            }
        p1_data = []
        p2_data = []
        num_right = 0

        for tester in training_data:
            logger.warn(tester[1:])
            p1 = self.parameters[1]['g'](tester[1:])
            p2 = self.parameters[2]['g'](tester[1:])

            if tester[0] == (1 if p1 > p2 else 2): num_right += 1
            logger.info("Pred: %i Actual: %i\tGot p1: %.3f and p2: %.3f" % (1 if p1 > p2 else 2, tester[0], p1, p2))
            p1_data.append(p1)
            p2_data.append(p2)
        logger.info("Got %i right out of %i" % (num_right, len(training_data)))

    def __repr__(self):
        return str(self)


class GaussianMLE(BayesMLE):
    pass
