from dataclasses import dataclass, replace
from typing import Tuple, Any, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix


@dataclass
class COOData:

    indices: ndarray
    data: ndarray
    shape: Tuple[int, ...]
    local_shape: Optional[Tuple[int, ...]]

    @staticmethod
    def _assemble_scipy_csr(
            indices: ndarray,
            data: ndarray,
            shape: Tuple[int, ...],
            local_shape: Optional[Tuple[int, ...]]
    ) -> csr_matrix:

        K = coo_matrix((data, (indices[0], indices[1])), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    def __radd__(self, other):
        return self.__add__(other)

    def inverse(self):
        """Invert each elemental matrix."""

        assert len(self.local_shape) == 2
        data = self.data.reshape(self.local_shape + (-1,), order='C')
        data = np.moveaxis(np.linalg.inv(np.moveaxis(data, -1, 0)), 0, -1)

        return replace(
            self,
            data=data.flatten('C'),
        )

    def __add__(self, other):

        if isinstance(other, int):
            return self

        return replace(
            self,
            indices=np.hstack((self.indices, other.indices)),
            data=np.hstack((self.data, other.data)),
            shape=tuple(max(self.shape[i],
                            other.shape[i]) for i in range(len(self.shape))),
            local_shape=None,
        )

    def tocsr(self) -> csr_matrix:
        """Return a sparse SciPy CSR matrix."""
        return self._assemble_scipy_csr(
            self.indices,
            self.data,
            self.shape,
            self.local_shape,
        )

    def toarray(self) -> ndarray:
        """Return a dense NumPy array."""
        if len(self.shape) == 1:
            return coo_matrix(
                (self.data, (self.indices[0], np.zeros_like(self.indices[0]))),
                shape=self.shape + (1,),
            ).toarray().T[0]
        elif len(self.shape) == 2:
            return self.tocsr().toarray()

        # slow implementation for testing N-tensors
        out = np.zeros(self.shape)
        for itr in range(self.indices.shape[1]):
            out[tuple(self.indices[:, itr])] += self.data[itr]
        return out

    def astuple(self):
        return self.indices, self.data, self.shape

    def todefault(self) -> Any:
        """Return the default data type.

        Scalar for 0-tensor, numpy array for 1-tensor, scipy csr matrix for
        2-tensor, self otherwise.

        """
        if len(self.shape) == 0:
            return self.data[0]
        elif len(self.shape) == 1:
            return self.toarray()
        elif len(self.shape) == 2:
            return self.tocsr()
        return self
