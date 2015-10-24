import numpy as np
import logging


logger = logging.getLogger(__name__)


class ParametricModel(object):
    def __init__(self):
        pass


class NormalModel(ParametricModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class LinearModel(ParametricModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
