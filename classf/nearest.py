import numpy as np
from .core import Model


class KNN(Model):
    def __init__(self, train, test):
        super().__init__(train, test)
