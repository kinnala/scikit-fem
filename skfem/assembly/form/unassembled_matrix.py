import numpy as np

from numpy import ndarray
from scipy.sparse import coo_matrix
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass
class UnassembledMatrix:

    data: ndarray
    rows: ndarray
    cols: ndarray
    shape: Tuple[int, int]

    def tocsr(self):

        K = coo_matrix(
            (self.data, (self.rows, self.cols)),
            shape=self.shape,
        )
        K.eliminate_zeros()
        return K.tocsr()

    def enforce(self, x, D):

        rows_mapping = np.ones(self.shape[0])
        rows_mapping[D] = 0

        return replace(
            self,
            data = np.concatenate((rows_mapping[self.rows] * self.data,
                                   np.ones(len(D)))),
            rows = np.concatenate((self.rows, D)),
            cols = np.concatenate((self.cols, D)),
        )
