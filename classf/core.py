import pickle
import numpy as np
import os
from typing import Iterable

__all__ = ['Classifier']


class Classifier(object):
    """
    Base class for all classifiers, defines the expected functions every classifier should have.
    """

    def __init__(self):
        """
        Create a Classifier.
        """
        pass

    def train(self, data: np.ndarray):
        """
        Train the classifier on a given data set.

        :param numpy.ndarray data: the data.
        """
        # the delegation chain stops here
        assert not hasattr(super(), 'train')

    def test(self, data: np.ndarray):
        """
        Test the classifier against a given data set.

        :param numpy.ndarray data: the data.
        """
        # the delegation chain stops here
        assert not hasattr(super(), 'test')

    def classify(self, data: np.ndarray) -> Iterable[int]:
        """
        Classify a data set, and return the results as an array of classes.

        :param numpy.ndarray data: the data
        :return: A list of classifications corresponding to the instances.
        :rtype [int]:
        """
        # the delegation chain stops here
        assert not hasattr(super(), 'classify')

    def save(self, location: str):
        """
        Pickle this Classifier.

        :param location: location to save pickled Classifier.
        """
        fpath = os.path.abspath(location)
        with open(fpath, 'wb') as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load(cls, location: str):
        """
        Load a pickled classifier.

        :param str location: location to load the classifier from.
        :return: Unpickled classifier
        :rtype Classifier:
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
    """
    classifier_name = classifier_name.lower()

    if classifier_name == 'mle':
        from .likelihood import BayesMLE
        return BayesMLE()

    if classifier_name == 'parzen' or classifier_name == 'p':
        raise NotImplementedError("Parzen Windows not implemented yet.")

    if classifier_name == 'knn':
        raise NotImplementedError("Nearest Neighbors not implemented yet!")

    raise KeyError("Couldn't find classifier with name %s." % classifier_name)

