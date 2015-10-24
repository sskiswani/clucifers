import numpy as np
import logging


logger = logging.getLogger(__name__)


class ParametricModel(object):
    def __init__(self):
        pass

class NormalModel(ParametricModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class UniformModel(ParametricModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class BernoulliModel(ParametricModel):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
