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

    def tolocal(self, basis=None):
        """Return an array of local finite element matrices.

        Parameters
        ----------
        basis
            Optionally, sum local facet matrices to form elemental matrices if
            the corresponding :class:`skfem.assembly.FacetBasis` is provided.

        """
        if self.local_shape is None:
            raise NotImplementedError("Cannot build local matrices if "
                                      "local_shape is not specified.")
        assert len(self.local_shape) == 2

        local = np.moveaxis(self.data.reshape(self.local_shape + (-1,),
                                              order='C'), -1, 0)
        if basis is not None:
            out = np.zeros((basis.mesh.nfacets,) + local.shape[1:])
            out[basis.find] = local
            local = np.sum(out[basis.mesh.t2f], axis=0)

        return local

    def fromlocal(self, local):
        """Reverse of :meth:`COOData.tolocal`."""
        return replace(
            self,
            data=np.moveaxis(local, 0, -1).flatten('C'),
        )

    def inverse(self):
        """Invert each elemental matrix."""
        return self.fromlocal(np.linalg.inv(self.tolocal()))

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
