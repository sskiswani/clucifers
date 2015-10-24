import numpy as np
import logging
from . import core
from . import distributions
from . import likelihood, nearest, parzen
from typing import Callable

_all_ = [
    'core',
    'distributions',
    'likelihood',
    'nearest',
    'parzen'
]


def run(classifier_name: str,
        training_data: str,
        converter: Callable[[str], np.ndarray]=None,
        testing_data: str=None,
        classify_data: str=None,
        verbose: bool=False) -> core.Classifier:
    """
    Used to simplify development mostly.

    :param str classifier_name: key used to fetch classifier
    :param str training_data: path to file containing classifier training data
    :param lambda converter: function that takes a file path and converts the corresponding
                             file into an array of feature vectors.
    :param str testing_data: path to file containing data used for testing.
    :param str classify_data: path to file containg data used for classification
    :param bool verbose: whether or not to output runtime information.

    :return: the classifer corresponding to classifier_name.
    :rtype core.Classifier:
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
        logging.debug('Logging level set to DEBUG')

    if converter is None: converter = np.loadtxt

    classifier = core.get_classifier(classifier_name)
    classifier.verbose = verbose

    np_train = converter(training_data)
    classifier.train(np_train)

    if testing_data is not None:
        np_test = converter(testing_data)
        classifier.test(np_test)

    if classify_data is not None:
        np_clsfy = converter(classify_data)
        classifier.classify(np_clsfy)

    return classifier
