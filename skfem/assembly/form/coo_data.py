from dataclasses import dataclass, replace
from typing import Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix


@dataclass
class COOData:

    data: ndarray
    rows: ndarray
    cols: ndarray
    shape: Tuple[int, int]

    @staticmethod
    def _assemble_scipy_csr(data: ndarray,
                            rows: ndarray,
                            cols: ndarray,
                            shape: Tuple[int, int]) -> csr_matrix:

        K = coo_matrix((data, (rows, cols)), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):

        if isinstance(other, int):
            return self

        return replace(
            self,
            data=np.concatenate((self.data, other.data)),
            rows=np.concatenate((self.rows, other.rows)),
            cols=np.concatenate((self.cols, other.cols)),
            shape=(max(self.shape[0], other.shape[0]),
                   max(self.shape[1], other.shape[1])),
        )

    def tocsr(self) -> csr_matrix:
        """Return a sparse SciPy CSR matrix."""
        return self._assemble_scipy_csr(
            self.data,
            self.rows,
            self.cols,
            self.shape,
        )

    def toarray(self) -> ndarray:
        """Return a dense NumPy array."""
        return coo_matrix((self.data, (self.rows, self.cols)),
                          shape=self.shape).toarray()
