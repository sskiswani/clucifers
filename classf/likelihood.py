import numpy as np
from .core import Classifier


class BayesMLE(Classifier):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, data: np.ndarray):
        pass

    def test(self, data: np.ndarray):
        pass

    def classify(self, data: np.ndarray):
        pass
