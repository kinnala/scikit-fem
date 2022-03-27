import logging
import sys

from dataclasses import dataclass, replace
from typing import Tuple, Any, Optional

import numpy as np
from numpy import ndarray

if "pyodide" in sys.modules:
    from scipy.sparse.coo import coo_matrix
else:
    from scipy.sparse import coo_matrix


logger = logging.getLogger(__name__)


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
    ):
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

    def tocsr(self):
        """Return a sparse SciPy CSR matrix."""
        return self._assemble_scipy_csr(
            self.indices,
            self.data,
            self.shape,
            self.local_shape,
        )

    def toarray(self) -> ndarray:
        """Return a dense numpy array."""
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
            return np.sum(self.data, axis=0)
        elif len(self.shape) == 1:
            return self.toarray()
        elif len(self.shape) == 2:
            return self.tocsr()
        return self

    def dot(self, x, D=None):
        """Matrix-vector product.

        Parameters
        ----------
        x
            The vector to multiply with.
        D
            Optionally, keep some DOFs unchanged.  An array of DOF indices.

        """
        y = self.data * x[self.indices[1]]
        z = np.zeros_like(x)
        np.add.at(z, self.indices[0], y)
        if D is not None:
            z[D] = x[D]
        return z

    def solve(self, b, D=None, tol=1e-10, maxiters=500):
        """Solve linear system using the conjugate gradient method.

        Parameters
        ----------
        b
            The right-hand side vector.
        D
            An optional array of Dirichlet DOF indices for which the fixed
            value is taken from ``b``.
        tol
            A tolerance for terminating the conjugate gradient method.
        maxiters
            The maximum number of iterations before forced termination.

        """
        x = b
        r = b - self.dot(x, D=D)
        p = r
        rsold = np.dot(r, r)
        for k in range(maxiters):
            Ap = self.dot(p, D=D)
            alpha = rsold / np.dot(p, Ap)
            x = x + alpha * p
            r = r - alpha * Ap
            rsnew = np.dot(r, r)
            if np.sqrt(rsnew) < tol:
                break
            p = r + (rsnew / rsold) * p
            rsold = rsnew
        if k == maxiters:
            logger.warning("Iterative solver did not converge.")
        return x
