import logging

import numpy as np
from typing import Callable

from .core import Classifier

logger = logging.getLogger(__name__)

KernelType = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def make_box_kernel(center: np.ndarray, width: np.ndarray) -> Callable[[np.ndarray], float]:
    volume = np.array([h ** d for d, h in enumerate(width)])
    hw = width / 2

    # noinspection PyTypeChecker
    def box_kernel(x: np.ndarray) -> float:
        return np.sum((np.abs(x - center) / hw <= 0.5) / volume)
        # return np.sum((np.abs(x - center) / hw) <= 0.5) / x.shape[0] / volume * x.shape[0]

    return box_kernel


def gauss_window(x: np.ndarray, mu: float, sigma: np.ndarray) -> np.ndarray:
    det_sig = np.linalg.det(sigma)

    denom = np.sqrt(((2 * np.pi) ** x.shape[0]) * det_sig)
    return np.exp(-0.5 * ((x - mu).T).dot(np.linalg.inv(sigma)).dot((x - mu))) / denom


def make_window(features: np.ndarray, key: str = 'box'):
    pass


class Kernel(object):
    def __init__(self, training_data: np.ndarray, centers: np.ndarray, extents: np.ndarray, **kwargs):
        self.centers = centers
        self.extents = extents

    @property
    def dimensions(self) -> int:
        return self.centers.shape[0]


class BoxKernel(Kernel):
    def __init__(self, training_data: np.ndarray, centers: np.ndarray, extents: np.ndarray, **kwargs):
        super().__init__(training_data, centers, extents, **kwargs)




class ParzenMLE(Classifier):
    def __init__(self, training_data: np.ndarray, kernel: str = "box", **kwargs):
        super().__init__(training_data, **kwargs)

        # Create the estimates for each class in the training_data
        classes = training_data[:, 0]
        self.parameters = {}

        for c in self.priors.keys():
            features = training_data[np.where(classes == c)][:, 1:]

            mu = np.sum(features, axis=0) / np.sum(classes == c)
            memo = np.matrix(features - mu)
            sigma = memo.T.dot(memo) / np.sum(classes == c)

            self.parameters[c] = {
                ''
            }

        if self.kernel is None:
            center = np.ones_like(training_data)
            volume = 0.5 * np.ones_like(training_data)
            self.kernel = make_box_kernel(center=center, width=volume)
