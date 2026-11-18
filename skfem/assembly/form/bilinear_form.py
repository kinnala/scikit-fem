import logging
from typing import Optional, Tuple
from threading import Thread
from itertools import product

import numpy as np
from numpy import ndarray

from ..basis import Basis
from .coo_data import COOData
from .form import Form, FormExtraParams


logger = logging.getLogger(__name__)


class BilinearForm(Form):
    """A bilinear form for finite element assembly.

    Bilinear forms are defined using functions that takes three arguments:
    trial function ``u``, test function ``v``, and a dictionary of additional
    parameters ``w``.

    >>> from skfem import BilinearForm, Basis, MeshTri, ElementTriP1
    >>> form = BilinearForm(lambda u, v, _: u * v)
    >>> form.assemble(Basis(MeshTri(), ElementTriP1())).toarray()
    array([[0.08333333, 0.04166667, 0.04166667, 0.        ],
           [0.04166667, 0.16666667, 0.08333333, 0.04166667],
           [0.04166667, 0.08333333, 0.16666667, 0.04166667],
           [0.        , 0.04166667, 0.04166667, 0.08333333]])

    Alternatively, you can use :class:`~skfem.assembly.BilinearForm` as a
    decorator:

    >>> @BilinearForm
    ... def form(u, v, _):
    ...     return u * v

    Inside the form definition, ``u`` and ``v`` are arrays containing the basis
    function values at quadrature points.  They also contain the values of
    the derivatives as attributes:

    >>> @BilinearForm
    ... def form(u, v, _):
    ...     # u.grad[0] is first derivative with respect to x, and so on
    ...     return u.grad[0] * v.grad[0] + u.grad[1] * v.grad[1]  # laplacian

    We suggest you to use helper functions from :mod:`skfem.helpers` to make
    the forms more readable:

    >>> from skfem.helpers import dot, grad
    >>> @BilinearForm
    ... def form(u, v, w):
    ...     return dot(grad(u), grad(v))

    """

    def _assemble(self,
                  ubasis: Basis,
                  vbasis: Optional[Basis] = None,
                  **kwargs) -> Tuple[ndarray,
                                     ndarray,
                                     Tuple[int, int],
                                     Tuple[int, int]]:

        if vbasis is None:
            vbasis = ubasis
        elif ubasis.X.shape[-1] != vbasis.X.shape[-1]:
            raise ValueError("Quadrature mismatch: trial and test functions "
                             "should have same number of integration points.")

        nt = ubasis.nelems
        dx = ubasis.dx
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })

        # initialize COO data structures
        data = np.zeros((ubasis.Nbfun, vbasis.Nbfun, nt), dtype=self.dtype)
        rowcol = np.empty((2, ubasis.Nbfun, vbasis.Nbfun, nt),
                          dtype=np.int32)
        rowcol[0] = vbasis.element_dofs[None, :, :]  # rows
        rowcol[1] = ubasis.element_dofs[:, None, :]  # cols

        # loop over the indices of local stiffness matrix
        if self.nthreads <= 0:
            for j in range(ubasis.Nbfun):
                for i in range(vbasis.Nbfun):
                    data[j, i, :] = self._kernel(
                        ubasis.basis[j],
                        vbasis.basis[i],
                        wdict,
                        dx,
                    )

        if self.nthreads > 0:
            # create indices for linear loop over local stiffness matrix
            indices = np.array(
                [[i, j] for j, i in product(range(ubasis.Nbfun),
                                            range(vbasis.Nbfun))]
            )

            # split local stiffness matrix elements to threads
            threads = [
                Thread(
                    target=self._threaded_kernel,
                    args=(data, ix, ubasis.basis, vbasis.basis, wdict, dx)
                ) for ix in np.array_split(indices, self.nthreads, axis=0)
            ]

            # start threads and wait for finishing
            for t in threads:
                t.start()
            for t in threads:
                t.join()

        return (
            rowcol.reshape(2, -1),
            data.reshape(-1),
            (vbasis.N, ubasis.N),
            (vbasis.Nbfun, ubasis.Nbfun),
        )

    def assemble(self, *args, **kwargs):
        """Assemble the bilinear form into a sparse matrix.

        If the basis is defined on a distributed mesh, see
        :func:`skfem.distributed.distribute`, returns a PETSc matrix.

        Parameters
        ----------
        ubasis
            The :class:`~skfem.assembly.Basis` for ``u``.
        vbasis
            Optionally, specify a different :class:`~skfem.assembly.Basis`
            for ``v``.
        **kwargs
            Any additional keyword arguments are appended to ``w``.

        """
        assert self.form is not None
        logger.info("Assembling '{}'.".format(self.form.__name__))
        dofs = self._distributed_dofs(*args)
        if dofs is not None:
            mat = (COOData(*self._assemble(*args, **kwargs))
                   ._todistributed(dofs))
        else:
            mat = COOData._assemble_scipy_csr(*self._assemble(*args,
                                                              **kwargs))
        logger.info("Assembling finished.")
        return mat

    def assemble_blocks(self, ubasis, vbasis=None, **kwargs):
        """Assemble the bilinear form and split it into blocks.

        The blocks correspond to the solution components, e.g., the
        elements of :class:`~skfem.element.ElementComposite`, the
        components of :class:`~skfem.element.ElementVector`, or the bases of
        :class:`~skfem.assembly.basis.CompositeBasis`.  The DOFs of each
        component are numbered as in
        :meth:`~skfem.assembly.CellBasis.split`.

        >>> from skfem import (BilinearForm, Basis, MeshTri, ElementTriP2,
        ...                    ElementTriP1, ElementVector)
        >>> from skfem.helpers import ddot, grad, div
        >>> @BilinearForm
        ... def stokes(u, p, v, q, w):
        ...     return ddot(grad(u), grad(v)) - div(u) * q - div(v) * p
        >>> basis = Basis(MeshTri(), ElementVector(ElementTriP2())
        ...               * ElementTriP1())
        >>> (A, B), (C, D) = stokes.assemble_blocks(basis)
        >>> A.shape, B.shape, C.shape, D.shape
        ((18, 18), (18, 4), (4, 18), (4, 4))

        Parameters
        ----------
        ubasis
            The :class:`~skfem.assembly.Basis` for ``u``.
        vbasis
            Optionally, specify a different :class:`~skfem.assembly.Basis`
            for ``v``.
        **kwargs
            Any additional keyword arguments are appended to ``w``.

        Returns
        -------
        list
            The sparse matrices ``blocks[i][j]`` of the ``i``'th component
            of ``v`` and the ``j``'th component of ``u``.

        """
        vbasis = ubasis if vbasis is None else vbasis
        if self._distributed_dofs(ubasis, vbasis) is not None:
            raise NotImplementedError("Distributed meshes not supported.")
        rows = self._components(vbasis)
        cols = self._components(ubasis)
        mat = self.assemble(ubasis, vbasis, **kwargs).tocsr()
        return [[mat[r][:, c] for c in cols] for r in rows]

    def elemental_blocks(self, ubasis, vbasis=None, **kwargs):
        """Like :meth:`assemble_blocks` but return the elemental matrices.

        The blocks are :class:`~skfem.assembly.form.COOData` objects
        supporting, e.g., :meth:`~skfem.assembly.form.COOData.tolocal`
        and :meth:`~skfem.assembly.form.COOData.inverse`.  Use
        ``tocsr()`` to obtain the sparse matrices of
        :meth:`assemble_blocks`.

        Parameters
        ----------
        ubasis
            The :class:`~skfem.assembly.Basis` for ``u``.
        vbasis
            Optionally, specify a different :class:`~skfem.assembly.Basis`
            for ``v``.
        **kwargs
            Any additional keyword arguments are appended to ``w``.

        Returns
        -------
        list
            The blocks ``blocks[i][j]`` of the ``i``'th component of ``v``
            and the ``j``'th component of ``u``.

        """
        vbasis = ubasis if vbasis is None else vbasis
        local = self.elemental(ubasis, vbasis, **kwargs).tolocal()
        vdofs, udofs = vbasis.element_dofs, ubasis.element_dofs
        blocks = []
        for vloc, vrenum, vn in self._local_components(vbasis):
            row = []
            for uloc, urenum, un in self._local_components(ubasis):
                block = local[:, vloc][:, :, uloc]
                rix = np.broadcast_to(vrenum[vdofs[vloc]].T[:, :, None],
                                      block.shape)
                cix = np.broadcast_to(urenum[udofs[uloc]].T[:, None, :],
                                      block.shape)
                row.append(COOData(
                    np.array([rix.T.flatten('C'), cix.T.flatten('C')]),
                    block.T.flatten('C'),
                    (vn, un),
                    (len(vloc), len(uloc)),
                ))
            blocks.append(row)
        return blocks

    def _kernel(self, u, v, w, dx):
        return self._integrate(self.form(*u, *v, w), dx)

    def _threaded_kernel(self, data, ix, ubasis, vbasis, wdict, dx):
        for ij in ix:
            i, j = ij
            data[j, i] = self._kernel(
                ubasis[j],
                vbasis[i],
                wdict,
                dx,
            )
