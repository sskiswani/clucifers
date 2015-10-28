import logging
import os
import pickle

import numpy as np
from typing import Iterable, Union, Optional

__all__ = [
    'Classifier',
]

logger = logging.getLogger(__name__)


def z_score(data: np.ndarray, mu: float = 0, sigma: float = 1, inplace: bool = True,
            out: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute the z-transform of the given data set.

    :param data: input
    :param mu:  mean of the data
    :param sigma: std dev.
    :param inplace:
    :param out: place to store the result.
    :return: z-transformed data.
    """
    return (data - mu) / sigma


def inv_zform(data: np.ndarray,
              out: Optional[np.ndarray] = None,
              clone: bool = True,
              sigma: float = 1,
              mu: float = 1) -> np.ndarray:
    if clone or out is None:
        out = data.copy()


class Classifier(object):
    """
    Base class for all classifiers, defines the expected functions every classifier should have.
    """

    def __init__(self, training_data: np.ndarray, **kwargs):
        self.training_data = training_data

        classes = training_data[:, 0]
        n = training_data.shape[0]
        self.priors = {a: (np.sum(classes == a) / n) for a in np.unique(classes)}

    def test(self, points: np.ndarray, labels: Optional[np.ndarray] = None):
        assert not hasattr(super(), 'test')

    def classify(self, data_set: np.ndarray):
        assert not hasattr(super(), 'classify')

    def iter_classify(self, data_set: np.ndarray):
        assert not hasattr(super(), 'iter_classify')

    @property
    def classes(self):
        return self.priors.keys()

    def save(self, location: str):
        """
        Pickle this Classifier.

        :param str location: location to save pickled Classifier.
        """
        fpath = os.path.abspath(location)
        with open(fpath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, location: str):
        """
        Load a pickled classifier.

        :param cls:
        :param str location: location to load the classifier from.
        :return: Unpickled classifier
        :rtype: Classifier
        """
        fname = os.path.abspath(location)
        with open(fname, 'rb') as f:
            attributes = pickle.load(f)

        obj = cls.__new__(cls)
        obj.__dict__.update(attributes)
        return obj

    def __str__(self):
        return "<Classfier\n" \
               "\tPriors: {!r}\n" \
               "\tTraining Data: {!r}\n" \
               ">".format(self.priors, self.training_data)


def get_classifier(classifier_name: str, training_data: Union[Iterable[np.ndarray], np.ndarray], **kwargs):
    """
    Get a classifier by a name keyword.

    :param str classifier_name: name of the classifier.
    :param training_data: data used to train the classifier
    :return: A classifier instance.
    :rtype: Classifier
    """
    classifier_name = classifier_name.lower()

    if classifier_name == 'mle':
        from .likelihood import BayesMLE
        return BayesMLE(training_data, **kwargs)

    if classifier_name == 'knn':
        from .nearest import NearestNeighbors
        return NearestNeighbors(training_data, **kwargs)

    if classifier_name == 'parzen' or classifier_name == 'p':
        raise NotImplementedError("Parzen window estimation not implemented yet.")

    raise KeyError("Couldn't find classifier with name %s." % classifier_name)
