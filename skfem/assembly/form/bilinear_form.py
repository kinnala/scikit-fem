from typing import Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import csr_matrix

from ..basis import Basis
from .coo_data import COOData
from .form import Form, FormExtraParams


class BilinearForm(Form):
    """A bilinear form for finite element assembly.

    Bilinear forms are defined using functions that takes three arguments:
    trial function ``u``, test function ``v``, and a dictionary of additional
    parameters ``w``.

    >>> from skfem import BilinearForm, InteriorBasis, MeshTri, ElementTriP1
    >>> form = BilinearForm(lambda u, v, w: u * v)
    >>> form.assemble(InteriorBasis(MeshTri(), ElementTriP1())).todense()
    matrix([[0.08333333, 0.04166667, 0.04166667, 0.        ],
            [0.04166667, 0.16666667, 0.08333333, 0.04166667],
            [0.04166667, 0.08333333, 0.16666667, 0.04166667],
            [0.        , 0.04166667, 0.04166667, 0.08333333]])

    Alternatively, you can use :class:`~skfem.assembly.BilinearForm` as a
    decorator:

    >>> @BilinearForm
    ... def form(u, v, w):
    ...     return u * v

    Inside the form definition, ``u`` and ``v`` are tuples containing the basis
    function values at quadrature points.  They also contain the values of
    the derivatives:

    >>> @BilinearForm
    ... def form(u, v, w):
    ...     # u[1][0] is first derivative with respect to x, and so on
    ...     return u[1][0] * v[1][0] + u[1][1] * v[1][1]  # laplacian

    In practice, we suggest you to use helper functions from
    :mod:`skfem.helpers` to make the forms readable:

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
                                     ndarray,
                                     Tuple[int, int]]:

        if vbasis is None:
            vbasis = ubasis
        elif ubasis.X.shape[1] != vbasis.X.shape[1]:
            raise ValueError("Quadrature mismatch: trial and test functions "
                             "should have same number of integration points.")

        nt = ubasis.nelems
        dx = ubasis.dx
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self.dictify(kwargs),
        })

        # initialize COO data structures
        sz = ubasis.Nbfun * vbasis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int64)
        cols = np.zeros(sz, dtype=np.int64)

        # loop over the indices of local stiffness matrix
        for j in range(ubasis.Nbfun):
            for i in range(vbasis.Nbfun):
                ixs = slice(nt * (vbasis.Nbfun * j + i),
                            nt * (vbasis.Nbfun * j + i + 1))
                rows[ixs] = vbasis.element_dofs[i]
                cols[ixs] = ubasis.element_dofs[j]
                data[ixs] = self._kernel(
                    ubasis.basis[j],
                    vbasis.basis[i],
                    wdict,
                    dx,
                )

        return data, rows, cols, (vbasis.N, ubasis.N)

    def coo_data(self, *args, **kwargs) -> COOData:
        return COOData(*self._assemble(*args, **kwargs))

    def assemble(self, *args, **kwargs) -> csr_matrix:
        """Assemble the bilinear form into a sparse matrix.

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
        return COOData._assemble_scipy_csr(*self._assemble(*args, **kwargs))

    def _kernel(self, u, v, w, dx):
        return np.sum(self.form(*u, *v, w) * dx, axis=1)
