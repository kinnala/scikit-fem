import numpy as np

from numpy import ndarray
from scipy.sparse import coo_matrix
from dataclasses import dataclass, replace
from typing import Tuple


@dataclass
class COOData:

    data: ndarray
    rows: ndarray
    cols: ndarray
    shape: Tuple[int, int]

    @staticmethod
    def _assemble_scipy_csr(data, rows, cols, shape):

        K = coo_matrix((data, (rows, cols)), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    def tocsr(self):

        return self._assemble_scipy_csr(
            self.data,
            self.rows,
            self.cols,
            self.shape,
        )

    def enforce(self, D):

        rows_mapping = np.ones(self.shape[0])
        rows_mapping[D] = 0

        return replace(
            self,
            data=np.concatenate((rows_mapping[self.rows] * self.data,
                                 np.ones(len(D)))),
            rows=np.concatenate((self.rows, D)),
            cols=np.concatenate((self.cols, D)),
        )
