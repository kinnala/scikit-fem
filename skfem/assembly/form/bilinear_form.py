from typing import Optional, Any

import numpy as np

from .form import Form, FormDict
from ..basis import Basis


class BilinearForm(Form):
    """A bilinear form for finite element assembly.

    Bilinear forms take three arguments: trial function `u`, test function `v`,
    and a dictionary of additional parameters `w`.

    >>> form = BilinearForm(lambda u, v, w: u * v)

    Alternatively, you can use :class:`~skfem.assembly.BilinearForm` as a
    decorator:

    >>> @BilinearForm
    ... def form(u, v, w):
    ...     return u * v

    Inside the form definition, `u` and `v` are tuples containing the basis
    function values at quadrature points.  They also contain the values of
    the derivatives:

    >>> @BilinearForm
    ... def form(u, v, w):
    ...     # u[1][0] is first derivative with respect to x, and so on
    ...     return u[1][0] * v[1][0] + u[1][1] * v[1][1]  # laplacian

    In practice, we suggest you to use helper functions:

    >>> @BilinearForm
    ... def form(u, v, w):
    ...     from skfem.helpers import dot, grad
    ...     return dot(grad(u), grad(v))

    """

    def assemble(self,
                 u_basis: Basis,
                 v_basis: Optional[Basis] = None,
                 **kwargs) -> Any:

        if v_basis is None:
            v_basis = u_basis
        elif u_basis.X.shape[1] != v_basis.X.shape[1]:
            raise ValueError("Quadrature mismatch: trial and test functions "
                             "should have same number of integration points.")

        nt = u_basis.nelems
        dx = u_basis.dx
        w = FormDict({**u_basis.default_parameters(), **self.dictify(kwargs)})

        # initialize COO data structures
        sz = u_basis.Nbfun * v_basis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        # loop over the indices of local stiffness matrix
        for j in range(u_basis.Nbfun):
            for i in range(v_basis.Nbfun):
                ixs = slice(nt * (v_basis.Nbfun * j + i),
                            nt * (v_basis.Nbfun * j + i + 1))
                rows[ixs] = v_basis.element_dofs[i]
                cols[ixs] = u_basis.element_dofs[j]
                data[ixs] = self._kernel(u_basis.basis[j], v_basis.basis[i], w, dx)

        # TODO: allow user to change, e.g. cuda or petsc
        return self._assemble_scipy_matrix(data, rows, cols, (v_basis.N, u_basis.N))

    def _kernel(self, u, v, w, dx):
        return np.sum(self.form(*u, *v, w) * dx, axis=1)
