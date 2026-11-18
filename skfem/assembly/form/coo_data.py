import warnings
import logging

from dataclasses import dataclass, replace
from typing import Tuple, Any, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix


logger = logging.getLogger(__name__)


@dataclass
class COOData:
    """Assembled finite element data in coordinate format.

    ``local_shape`` lists the number of local basis functions in the same
    order as the global ``shape``, e.g., ``(Nbfun_v, Nbfun_u)`` for a
    bilinear form whose rows correspond to the test function ``v`` and
    columns to the trial function ``u``.  The entries of ``data`` are stored
    with the element index running fastest and the local indices in reverse
    order, i.e. ``data`` is a flattened array of shape
    ``local_shape[::-1] + (nelems,)``.

    """

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

    def __array__(self, dtype=None, copy=None):
        # allow passing a list of COOData objects to, e.g., np.concatenate
        out = self.toarray()
        if dtype is not None:
            out = out.astype(dtype, copy=False)
        return out

    def tolocal(self, basis=None):
        """Return an array of local finite element matrices.

        The result has the shape ``(nelems,) + local_shape`` and is indexed
        like the global matrix, e.g., ``out[e, i, j]`` is the entry related
        to the ``i``'th test function and the ``j``'th trial function in
        element ``e``.

        Parameters
        ----------
        basis
            Optionally, the basis used in the assembly, e.g.,
            :class:`~skfem.assembly.FacetBasis`.  The local matrices are then
            summed to the elements of the mesh they are related to, i.e.
            ``basis.tind``, which makes sense if the local basis functions
            are those of the element, e.g., the traces of the basis
            functions from one side of the facets.  The result has the shape
            ``(basis.mesh.nelements,) + local_shape``.

        """
        if self.local_shape is None:
            raise NotImplementedError("Cannot build local matrices if "
                                      "local_shape is not specified.")

        # data is laid out as local_shape[::-1] + (nelems,); reverse all axes
        local = self.data.reshape(self.local_shape[::-1] + (-1,)).T
        if basis is not None:
            tind = basis.tind
            if tind is None:
                tind = np.arange(basis.mesh.nelements)
            if len(tind) != local.shape[0]:
                raise ValueError("The basis does not correspond to the "
                                 "local matrices.")
            out = np.zeros((basis.mesh.nelements,) + local.shape[1:],
                           dtype=local.dtype)
            np.add.at(out, tind, local)
            local = out

        return local

    def fromlocal(self, local):
        """Reverse of :meth:`COOData.tolocal`."""
        return replace(
            self,
            data=np.asarray(local).T.flatten('C'),
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
        out = np.zeros(self.shape, dtype=self.data.dtype)
        for itr in range(self.indices.shape[1]):
            out[tuple(self.indices[:, itr])] += self.data[itr]
        return out

    def topetsc(self, dofs=None):
        """Convert to PETSc matrix or vector.

        Parameters
        ----------
        dofs
            Deprecated and due for removal in 14.0.0.  If given, then the
            matrix will be distributed ("mpiaij" in PETSc jargon).
            Otherwise, a local matrix is returned.  Use
            :meth:`skfem.mesh.Mesh.distributed` instead: then the forms
            assemble directly into distributed PETSc matrices.

        """
        if dofs is not None:
            warnings.warn("COOData.topetsc(dofs) is deprecated and will be "
                          "removed in 14.0.0. Use Mesh.distributed() and "
                          "Form.assemble instead.", DeprecationWarning,
                          stacklevel=2)
        return self._topetsc(dofs)

    def _topetsc(self, dofs=None):
        import petsc4py.PETSc as petsc

        if len(self.shape) == 1:
            f = self.toarray()
            locvec = petsc.Vec().createWithArray(f, comm=petsc.COMM_SELF)
            if dofs is None:
                return locvec

            comm = dofs._comm
            vec = petsc.Vec().create(comm=comm)
            vec.setSizes(dofs._nglob)
            vec.setType(petsc.Vec.Type.MPI)
            scat = petsc.Scatter().create(
                locvec,
                None,
                vec,
                dofs._iset,
            )
            scat.scatter(locvec, vec, addv=petsc.InsertMode.ADD)

            return vec

        elif len(self.shape) == 2:
            K = self.tocsr()
            locmat = petsc.Mat().createAIJ(size=K.shape,
                                           csr=(K.indptr,
                                                K.indices,
                                                K.data),
                                           comm=petsc.COMM_SELF)
            if dofs is None:
                return locmat

            comm = dofs._comm
            mat = petsc.Mat().create(comm=comm)
            mat.setSizes((dofs._nglob,
                          dofs._nglob), bsize=1)
            mat.setType(petsc.Mat.Type.IS)
            mat.setLGMap(dofs._lgmap)
            mat.setISLocalMat(locmat)
            mat.assemble()
            mat.convert("mpiaij")
            # the mapping is lost in the conversion; it is used, e.g.,
            # in skfem.utils.enforce to map the local DOF indices
            mat.setLGMap(dofs._lgmap)

            return mat

        raise NotImplementedError

    def _todistributed(self, dofs) -> Any:
        """Assemble over all processes of a distributed mesh.

        Scalar for 0-tensor (summed over the processes), PETSc vector for
        1-tensor and PETSc matrix for 2-tensor.

        """
        if len(self.shape) == 0:
            if dofs._comm.size == 1:
                return self.todefault()
            return dofs._comm.tompi4py().allreduce(self.todefault())
        return self._topetsc(dofs)

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
        z = np.zeros_like(x, dtype=y.dtype)
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
        if np.sqrt(rsold) < tol:
            return x
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
        else:
            logger.warning("Iterative solver did not converge.")
        return x
