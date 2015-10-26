import numpy as np
import logging
from typing import Callable, Optional, Iterable

from . import core
from . import likelihood, nearest, parzen

_all_ = [
    'core',
    'likelihood',
    'nearest',
    'parzen'
]

def parse_file(fpath: str) -> np.ndarray:
    """
    :param str fpath: path to the file containing the data.

    :return: List of ndarrays.
    :rtype np.ndarray:
    """
    return np.genfromtxt(fpath)


def run(classifier_name: str,
        training_data: str,
        converter: Callable[[str], Iterable[np.ndarray]]=parse_file,
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

    np_train = converter(training_data)
    classifier = core.get_classifier(classifier_name, np_train)

    if testing_data is not None:
        np_test = converter(testing_data)
        if not isinstance(classifier, nearest.NearestNeighbors):
            classifier.test(testing_data)
            return

        results = [classifier.test(np_test, alt_k=i) for i in range(1, 20)]

        import matplotlib.pyplot as plt
        plt.plot(range(1, 20), results)
        plt.xlabel('k')
        plt.ylabel('Accuracy')
        plt.xticks(range(1,20))
        plt.grid()
        plt.show()


    if classify_data is not None:
        np_clsfy = converter(classify_data)
        # classifier.classify(np_clsfy)

    return classifier
