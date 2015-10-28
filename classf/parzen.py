import logging

import numpy as np
from typing import Callable

from .core import Classifier

logger = logging.getLogger(__name__)

KernelType = Callable[[np.ndarray], float]
TWOPI_SQRT = np.sqrt(2 * np.pi)


# noinspection PyTypeChecker
def box_window(x: np.ndarray, center: np.ndarray, width: np.ndarray) -> float:
    return np.sum((np.abs((x - center) / width) <= 0.5)) / (width[0] ** width.shape[0])


def make_box_kernel(center: np.ndarray, width: np.ndarray) -> Callable[[np.ndarray], float]:
    volume = np.array([h ** d for d, h in enumerate(width)])

    # noinspection PyTypeChecker
    def box_kernel(x: np.ndarray) -> float:
        return np.sum((np.abs(x - center) / width <= 0.5) / volume)

    return box_kernel


def make_gauss_kernel(center: np.ndarray, width: np.ndarray, ) -> Callable[[np.ndarray], float]:
    volume = np.array([h ** d for d, h in enumerate(width)])
    consts = np.array([1. / ((TWOPI_SQRT * h) ** d) for d, h in enumerate(width)])

    def gauss_window(x: np.ndarray) -> float:
        return np.sum((np.abs(np.exp(-0.5 * (((x - center) / width) ** 2)) / consts) <= 0.5) / volume)

    return gauss_window


class ParzenParams(object):
    def __init__(self, label, features: np.ndarray, **kwargs):
        self.label = label
        self.count = features.shape[0]
        self.mean = np.mean(features, axis=0, dtype=np.float32)
        self.var = np.std(features, axis=0, dtype=np.float32)

    def p(self, x: np.ndarray, kernel: KernelType) -> float:
        return kernel((x - self.mean) / self.var) / self.count

    def __repr__(self):
        return repr(self.__dict__)


class ParzenWindows(Classifier):
    def __init__(self,
                 training_data: np.ndarray,
                 center: np.ndarray = None,
                 width: np.ndarray = None,
                 kernel_id: str = 'box',
                 **kwargs):
        super().__init__(training_data, **kwargs)
        self.center = center
        self.width = width

        if self.center is None:
            self.center = np.zeros_like(training_data[0, 1:])

        if self.width is None:
            self.width = np.ones_like(training_data[0, 1:])

        if kernel_id == 'gauss':
            self.kernel = make_gauss_kernel(self.center, self.width)
            self._kern_id = kernel_id
        else:
            self.kernel = make_box_kernel(self.center, self.width)
            self._kern_id = 'box'

        classes = training_data[:, 0]
        self.parameters = {}

        for label in self.priors.keys():
            features = training_data[np.where(classes == label)][:, 1:]
            best = -1
            best_params = None
            all_p = []
            for i in range(features.shape[0]):
                data = np.delete(features, i, axis=0)

                params = ParzenParams(label, data)
                p = params.p(features[i], self.kernel)
                # print('got %f vs %f' % (best, p))
                all_p.append(p)
                if best < p:
                    best = p
                    best_params = params

            if best <= 0:
                exit()

            self.parameters[label] = ParzenParams(label, data)

    def test(self, points: np.ndarray, labels: np.ndarray):
        # return self.test2(points, labels)
        num_right = 0

        for i, pred in enumerate(self.classify(points)):
            if labels[i] == pred:
                num_right += 1

        total = points.shape[0]
        acc = 100. * (num_right / total)
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (num_right, total, acc))

        return num_right, acc

        # best_right = 0
        #
        # for j in range(points.shape[0]):
        #     data = np.delete(points, j, axis=0)
        #     labs = np.delete(labels, j, axis=0)
        #
        #     num_right = 0
        #     for i, pred in enumerate(self.classify(data)):
        #         if labs[i] == pred:
        #             num_right += 1
        #
        #     if num_right > best_right:
        #         best_right = num_right
        #
        # total = points.shape[0]
        # acc = 100. * (best_right / total)
        # logger.info('Got %i right out of %i (%.2f%% accuracy)' % (best_right, total, acc))
        #
        # return best_right, acc

    def classify(self, data_set: np.ndarray):
        return list(self.iter_classify(data_set))

    def iter_classify(self, data_set: np.ndarray):
        classes = list(self.priors.keys())
        for x in data_set:
            yield classes[np.argmax([self.parameters[c].p(x, self.kernel) for c in classes])]

    def test2(self, points: np.ndarray, labels: np.ndarray):
        num_right = 0
        classes = list(self.priors.keys())

        for x, actual in zip(points, labels):
            probs = [self.parameters[c].p(x, self.kernel) for c in classes]
            pred = classes[np.argmax(probs)]
            logger.info('\t(window: %s) probs: %r' % (str(self.width), probs))
            logger.info('\tp sum: %r' % (np.sum(probs)))

            exit()

            if actual == pred:
                num_right += 1
        total = points.shape[0]
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (num_right, total, 100. * num_right / total))
        return num_right, 100. * num_right / total

    def set_width(self, width: np.ndarray):
        self.width = width
        if self._kern_id == 'gauss':
            self.kernel = make_gauss_kernel(self.center, self.width)
        else:
            self.kernel = make_box_kernel(self.center, self.width)
