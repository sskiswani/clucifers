import numpy as np
from .core import Model


class MLE(Model):
    def __init__(self, train, test, verbose=False):
        super().__init__(train, test, verbose)
