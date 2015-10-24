import pickle
import numpy as np
import os

def dataToArray(file):
    pass


class Model(object):
    def __init__(self, train, test, verbose=False):
        self.train = train
        self.test = test
        self.verbose = verbose

    def save(self, location: str):
        fpath = os.path.abspath(location)
        with open(fpath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, location: str):
        fname = os.path.abspath(location)
        with open(fname, 'rb') as f:
            attributes = pickle.load(f)

        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj
