import logging
import numpy as np
from typing import List, Optional, Callable, Union, Iterable
from collections import Counter
from .core import Classifier

logger = logging.getLogger(__name__)

class kdTree(object):
    def __init__(self, data:np.ndarray, depth:int = 0, min_bin:int=10):
        self.d = data.shape[-1]
        self.axis = (depth % self.d) + 1
        if self.axis == 0:
            raise NameError("uhoh")

        # This is a leaf node.
        if data.shape[0] < min_bin * 2:
            self.data = data
            return

        sorted = data[data[:, self.axis].argsort()]
        self.split = np.median(data, axis=0)[self.axis]

        median = data.shape[0] // 2
        self.left = kdTree(sorted[:median], depth+1, min_bin)
        self.right = kdTree(sorted[median:], depth+1, min_bin)


    def get_neighbors(self,
                      point:np.ndarray,
                      d:Callable[[np.ndarray, np.ndarray], float],
                      k:int=1,
                      result:Optional[List[np.ndarray]]=None) -> List[np.ndarray]:
        """

        :param point:
        :param callable d:
        :param int k:
        :param result:
        :return:
        """
        if result is None: result = []
        if self.is_leaf:
            arg_values = np.argsort([d(x, point) for x in self.data])
            result.extend(self.data[arg_values[:min(k,len(arg_values))]])
            return result

        if point[self.axis] <= self.split:
            norder = (self.left, self.right)
        else:
            norder = (self.right, self.left)

        norder[0].get_neighbors(point, d, k, result)
        if len(result) < k:
            norder[1].get_neighbors(point, d, k - len(result), result)

        return result

    @property
    def is_leaf(self) -> bool:
        return hasattr(self, 'data')

class NearestNeighbors(Classifier):
    def __init__(self, training_data:np.ndarray, k:int=5, **kwargs):
        super().__init__(training_data, **kwargs)
        self.tree = kdTree(training_data)
        self.k = k

    def test(self, testing_data: Union[Iterable[np.ndarray], np.ndarray], alt_k:Optional[int]=None, d:Optional[Callable[[np.ndarray, np.ndarray], float]]=None):
        if alt_k is None: alt_k = self.k
        if d is None: d = lambda x, y: np.linalg.norm(y[1:] - x[1:])
        num_right = 0
        for x in testing_data:
            neighbors = self.tree.get_neighbors(x, d, alt_k)
            ncount = Counter()
            ncount.update([x[0] for x in neighbors])
            if x[0] == ncount.most_common()[0][0]: num_right += 1
        logger.info('Got %i right out of %i (%.2f%% accuracy)' % (
        num_right, len(testing_data), (num_right / len(testing_data)) * 100))
        return num_right

