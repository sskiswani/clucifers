import numpy as np
import logging
from typing import Iterable, Union
from .core import Classifier

logger = logging.getLogger(__name__)

class BayesMLE(Classifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def train(self, data: Union[Iterable[np.ndarray], np.ndarray]):

        self.trained = np.array(self.training_data)


    def __repr__(self):
        return str(self)
