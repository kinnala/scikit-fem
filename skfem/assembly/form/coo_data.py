from dataclasses import dataclass
from typing import Tuple

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

    def toarray(self) -> ndarray:
        """Return a dense NumPy array."""
        return coo_matrix((self.data, (self.rows, self.cols)),
                          shape=self.shape).toarray()
