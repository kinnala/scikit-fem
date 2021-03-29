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
        """Return a sparse SciPy CSR matrix."""
        return self._assemble_scipy_csr(
            self.data,
            self.rows,
            self.cols,
            self.shape,
        )

    def todense(self) -> ndarray:
        """Return a dense NumPy array."""
        return coo_matrix((self.data, (self.rows, self.cols)),
                          shape=self.shape).todense()

    def enforce(self, D: ndarray, diag: float = 1.):
        """Enforce an essential BC by setting rows and diagonals to 0 and 1.

        Parameters
        ----------
        D
            An array of (Dirichlet) degrees-of-freedom to enforce.
        diag
            The value at the diagonals which is by default 1.

        """
        rows_mapping = np.ones(self.shape[0])
        rows_mapping[D] = 0

        data = rows_mapping[self.rows] * self.data
        rows = self.rows
        cols = self.cols

        if diag != 0:
            data = np.concatenate((data, np.zeros(len(D)) + diag))
            rows = np.concatenate((rows, D))
            cols = np.concatenate((cols, D))

        return replace(self, data=data, rows=rows, cols=cols)
