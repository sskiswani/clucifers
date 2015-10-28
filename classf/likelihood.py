import logging

import numpy as np
from typing import Iterable, Union, Optional

from .core import Classifier

logger = logging.getLogger(__name__)


def make_discriminant(mu, sigma, prior):
    det_sig = np.linalg.det(sigma)
    if det_sig == 0:
        invSig = np.ones(sigma.shape)
        det_sig = 1
    else:
        invSig = np.linalg.inv(sigma)

    return lambda x: prior * np.exp(-0.5 * ((x - mu).T).dot(invSig).dot((x - mu))) / np.sqrt(
        ((2 * np.pi) ** x.shape[0]) * det_sig)


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
        from pprint import pprint
        np.set_printoptions(precision=3, suppress=True)
        pprint(self.parameters)

    def test(self, points: np.ndarray, labels: Optional[np.ndarray] = None):
        num_right = 0

        for i, pred in enumerate(self.classify(points)):
            if labels[i] == pred:
                num_right += 1

        total = points.shape[0]
        acc = 100. * (num_right / total)
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (num_right, total, 100. * num_right / total))

        return num_right, acc

    def classify(self, data_set: np.ndarray):
        return list(self.iter_classify(data_set))

    def iter_classify(self, data_set: np.ndarray):
        classes = list(self.priors.keys())
        for x in data_set:
            probablities = []

            for c in classes:
                probablities.append(self.parameters[c]['g'](x))

            yield classes[np.argmax(probablities)]

    def test_old(self, testing_data: Union[Iterable[np.ndarray], np.ndarray]):
        classes = list(self.priors.keys())
        num_right = 0

        for x in testing_data:
            probablities = []
            for c in classes:
                probablities.append(self.parameters[c]['g'](x[1:]))
            pred = classes[np.argmax(probablities)]
            if x[0] == pred: num_right += 1

        total = testing_data.shape[0]
        acc = 100. * (num_right / total)
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (num_right, total, 100. * num_right / total))

        return num_right, acc

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
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (
            num_right, len(testing_data), (num_right / len(testing_data)) * 100))

    def __repr__(self):
        return str(self)


class UniformMLE(Classifier):
    pass
