from . import core
from . import likelihood, nearest, parzen

_all_ = [
    'likelihood',
    'nearest',
    'parzen'
]


def parse_file(file_handle):
    pass


def run(clsf, train, test, verbose, converter=parse_file):
    print("got clsf: {!r} train: {!r}, test: {!r}, v: {!r}".format(clsf, train, test, verbose))
    clsf = clsf.lower()

    np_train = converter(train)
    np_test = converter(test)

    if clsf == 'mle':
        return likelihood.MLE(np_train, np_test, verbose)
    if clsf == 'parzen' or clsf == 'p':
        raise NotImplementedError("Parzen Windows not implemented yet.")
    if clsf == 'knn':
        raise NotImplementedError("Nearest Neighbors not implemented yet!")
