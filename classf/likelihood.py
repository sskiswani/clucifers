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

        p1_data = []
        p2_data = []
        num_right = 0
        for tester in training_data:
            p1 = self.parameters[1]['g'](tester[1:])
            p2 = self.parameters[2]['g'](tester[1:])
            if tester[0] == (1 if p1 > p2 else 2): num_right += 1
            logger.info("Pred: %i Actual: %i\tGot p1: %.3f and p2: %.3f" % (1 if p1 > p2 else 2, tester[0], p1, p2))
            p1_data.append(p1)
            p2_data.append(p2)
        logger.info("Got %i right out of %i" % (num_right, len(training_data)))
        p1_data = np.array(p1_data)
        p2_data = np.array(p2_data)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 1)
        ax.fill_between(range(len(p1_data)), p1_data, p2_data, where=p1_data > p2_data, facecolor='green', interpolate=True)
        ax.fill_between(range(len(p1_data)), p1_data, p2_data, where=p1_data < p2_data, facecolor='blue', interpolate=True)

        # ax.fill_between(range(len(p1_data)), p2_data, p1_data, facecolor='red')

        plt.show()

    def __repr__(self):
        return str(self)


class GaussianMLE(BayesMLE):
    pass
