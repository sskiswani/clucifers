import numpy as np
from .core import Model


class Parzen(Model):
    def __init__(self, train, test):
        super().__init__(train, test)
