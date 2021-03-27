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

    def tocsr(self) -> csr_matrix:

        return self._assemble_scipy_csr(
            self.data,
            self.rows,
            self.cols,
            self.shape,
        )

    def enforce(self, D):
        """Enforce an essential BC by setting rows and diagonals to 0 and 1.

        Parameters
        ----------
        D
            A list of (Dirichlet) degrees-of-freedom to enforce.

        """
        rows_mapping = np.ones(self.shape[0])
        rows_mapping[D] = 0

        return replace(
            self,
            data=np.concatenate((rows_mapping[self.rows] * self.data,
                                 np.ones(len(D)))),
            rows=np.concatenate((self.rows, D)),
            cols=np.concatenate((self.cols, D)),
        )
