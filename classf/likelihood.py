import numpy as np
import logging
from typing import Iterable, Union
from .core import Classifier

logger = logging.getLogger(__name__)

def make_discriminant(mu, sigma, prior):
    det_sig = np.linalg.det(sigma)
    return lambda x: prior * np.exp(-0.5 * ((x - mu).T).dot(np.linalg.inv(sigma)).dot((x - mu))) / np.sqrt(((2*np.pi)**x.shape[0])*det_sig)

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

            self.parameters[c] = {
                'mu': mu,
                'sigma': sigma,
                'g': make_discriminant(mu.copy(), sigma.copy(), self.priors[c])
            }

        # p1_data = []
        # p2_data = []
        # num_right = 0
        #
        # for tester in training_data:
        #     logger.warn(tester[1:])
        #     p1 = self.parameters[1]['g'](tester[1:])
        #     p2 = self.parameters[2]['g'](tester[1:])
        #     p3 = self.parameters[3]['g'](tester[1:])
        #     test = [p1, p2, p3]
        #
        #     p = (np.argmax([p1,p2,p3]) +1)
        #     if tester[0] == p: num_right += 1
        #     logger.info("Pred: %i Actual: %i\tGot p1: %.3f | p2: %.3f | p3: %.3f" % (p, tester[0], p1, p2, p3))
        #
        # logger.info("Got %i right out of %i" % (num_right, len(training_data)))

    def test(self, testing_data: Union[Iterable[np.ndarray], np.ndarray]):
        classes = list(self.priors.keys())
        num_right = 0
        for x in testing_data:
            probablities = []
            for c in classes:
                probablities.append(self.parameters[c]['g'](x[1:]))
            pred = classes[np.argmax(probablities)]
            if x[0] == pred: num_right += 1
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (num_right, len(testing_data), (num_right / len(testing_data))*100))

    def __repr__(self):
        return str(self)


class GaussianMLE(Classifier):
    def __init__(self, training_data: Union[Iterable[np.ndarray], np.ndarray], **kwargs):
        super().__init__(training_data, **kwargs)

        classes = training_data[:, 0]
        self.parameters = {}
        for c in self.priors.keys():
            features = training_data[np.where(classes == c)][:, 1:]

            mu = np.sum(features, axis=0) / np.sum(classes == c)

            memo = np.matrix(features - mu)
            sigma = memo.T.dot(memo) / np.sum(classes == c)

            self.parameters[c] = {
                'mu': mu,
                'sigma': sigma,
                'g': make_discriminant(mu.copy(), sigma.copy(), self.priors[c])
            }

    def test(self, testing_data: Union[Iterable[np.ndarray], np.ndarray]):
        classes = list(self.priors.keys())
        num_right = 0
        for x in testing_data:
            probablities = []
            for c in classes:
                probablities.append(self.parameters[c]['g'](x[1:]))
            pred = classes[np.argmax(probablities)]
            if x[0] == pred: num_right += 1
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (num_right, len(testing_data), (num_right / len(testing_data))*100))

    def __repr__(self):
        return str(self)

class UniformMLE(Classifier):
    pass
