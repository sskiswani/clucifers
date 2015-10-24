import numpy as np
import logging
from typing import Callable, Optional, Iterable

from . import core
from . import distributions
from . import likelihood, nearest, parzen

_all_ = [
    'core',
    'distributions',
    'likelihood',
    'nearest',
    'parzen'
]

SomeType = np.ndarray

def parse_file(fpath: str) -> np.ndarray:
    """
    :param str fpath: path to the file containing the data.

    :return: List of ndarrays.
    :rtype [ndarray]:
    """
    return np.genfromtxt(fpath)
    # with open(os.path.abspath(fpath), 'r') as f:
    #     result = [np.fromstring(line, sep=' ') for line in f]
    # return result


def run(classifier_name: str,
        training_data: str,
        converter: Callable[[str], Iterable[SomeType]]=parse_file,
        testing_data: Optional[str]=None,
        classify_data: Optional[str]=None,
        verbose: Optional[bool]=False) -> core.Classifier:
    """
    Used to simplify development mostly.

    :param str classifier_name: key used to fetch classifier
    :param str training_data: path to file containing classifier training data
    :param (str) -> list[np.ndarray] converter: function that takes a file path and converts the corresponding
                             file into an array of feature vectors.
    :param str testing_data: path to file containing data used for testing.
    :param str classify_data: path to file containg data used for classification
    :param bool verbose: whether or not to output runtime information.

    :return core.Classifier: the classifer corresponding to classifier_name.
    """
    if verbose == True:
        logging.basicConfig(level=logging.DEBUG)
        logging.info('Logging level set to DEBUG')

    classifier = core.get_classifier(classifier_name)

    np_train = converter(training_data)
    classifier.train(np_train)

    classifier.train(np.array([1, 2, 3, 4, 5]))
    print(repr(classifier.trained))

    if testing_data is not None:
        np_test = converter(testing_data)
        # classifier.test(np_test)

    if classify_data is not None:
        np_clsfy = converter(classify_data)
        # classifier.classify(np_clsfy)

    return classifier
