import pickle
import logging
import os
import numpy as np
from typing import Iterable, Union

__all__ = [
    'Classifier'
]

logger = logging.getLogger(__name__)

class Classifier(object):
    """
    Base class for all classifiers, defines the expected functions every classifier should have.
    """
    def __init__(self, **kwargs):
        self.training_data = []
        self.priors = {}

    def add_training_data(self, data: Union[Iterable[np.ndarray], np.ndarray]):
        """
        Append training data.

        :param data: Training data.
        """
        raise NotImplementedError("mehh...")
        # if hasattr(data, 'shape'):
        #     if len(data.shape) == 1:
        #         self.training_data.append(data)
        #     elif len(data.shape) == 2:
        #         self.training_data.extend(data)
        #     else:
        #         raise TypeError("Improperly shaped data (shape = %s)" % repr(data.shape))

    def train(self, data: Union[Iterable[np.ndarray], np.ndarray]):
        """
        Train the classifier on a given data set.

        :param data: the data to use for training.
        """
        assert not hasattr(super(), 'train')

    def __str__(self):
        return "<Classfier\n" \
               "\tPriors: {!r}\n" \
               "\tTraining Data: {!r}\n" \
               ">".format(self.priors, self.training_data)

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


def get_classifier(classifier_name: str) -> Classifier:
    """
    Get a classifier by a name keyword.

    :param str classifier_name: name of the classifier.
    :return: A classifier instance.
    :rtype: Classifier
    """
    classifier_name = classifier_name.lower()

    if classifier_name == 'mle':
        from .likelihood import BayesMLE
        return BayesMLE()

    if classifier_name == 'parzen' or classifier_name == 'p':
        raise NotImplementedError("Parzen window estimation not implemented yet.")

    if classifier_name == 'knn':
        raise NotImplementedError("Nearest Neighbors not implemented yet!")

    raise KeyError("Couldn't find classifier with name %s." % classifier_name)

