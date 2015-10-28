import logging

import matplotlib.pyplot as plt
import numpy as np
from typing import Callable, Optional

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
        converter: Callable[[str], np.ndarray] = parse_file,
        testing_data: Optional[str] = None,
        classify_data: Optional[str] = None,
        verbose: bool = False) -> core.Classifier:
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

        if isinstance(classifier, nearest.NearestNeighbors):
            results = [classifier.test(np_test[:, 1:], np_test[:, 0], alt_k=i) for i in range(1, 20)]

            plt.plot(range(1, 20), results)
            plt.xlabel('k')
            plt.ylabel('Accuracy')
            plt.xticks(range(1, 20))
            plt.grid()
            plt.show()
        elif isinstance(classifier, parzen.ParzenWindows):
            results = []
            all_widths = np.linspace(0.1, 10, num=100)
            ones = np.ones_like(np_test[0, 1:])

            for width in all_widths:
                # new_classf = parzen.ParzenWindows(np_train, ones * width)
                new_classf = classifier
                new_classf.set_width(ones * width)
                results.append(new_classf.test(np_test[:, 1:], np_test[:, 0])[1])

            plt.plot(all_widths, results)
            plt.xlabel('Window width')
            plt.ylabel('Classification accuracy')
            plt.xticks(range(1, np.max(all_widths).astype(np.int)))
            plt.ylim((0, 100))
            plt.grid()
            plt.show()
        else:
            results = classifier.test(np_test[:, 1:], np_test[:, 0])

    if classify_data is not None:
        np_clsfy = converter(classify_data)
        # classifier.classify(np_clsfy)

    return classifier
