from dataclasses import dataclass, replace
from typing import Tuple, Any

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix


@dataclass
class COOData:

    indices: ndarray
    data: ndarray
    shape: Tuple[int, ...]

    @staticmethod
    def _assemble_scipy_csr(indices: ndarray,
                            data: ndarray,
                            shape: Tuple[int, int]) -> csr_matrix:

        K = coo_matrix((data, (indices[0], indices[1])), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    def __radd__(self, other):
        return self.__add__(other)

    def __add__(self, other):

        if isinstance(other, int):
            return self

        return replace(
            self,
            indices=np.hstack((self.indices, other.indices)),
            data=np.hstack((self.data, other.data)),
            shape=tuple(max(self.shape[i], other.shape[i]) for i in range(len(self.shape))),
        )

    def tocsr(self) -> csr_matrix:
        """Return a sparse SciPy CSR matrix."""
        return self._assemble_scipy_csr(
            self.indices,
            self.data,
            self.shape,
        )

    def toarray(self) -> ndarray:
        """Return a dense NumPy array."""
        return coo_matrix(
            (self.data, (self.indices[0], np.zeros_like(self.indices[0]))),
            shape=self.shape + (1,),
        ).toarray().T[0]

    def todefault(self) -> Any:

        if len(self.shape) == 0:
            return self.data[0]
        elif len(self.shape) == 1:
            return self.toarray()
        elif len(self.shape) == 2:
            return self.tocsr()
        return self
