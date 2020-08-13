from typing import Callable, Optional, Any

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
                 u: Basis,
                 v: Optional[Basis] = None,
                 **kwargs) -> Any:

        if v is None:
            v = u
        elif u.X.shape[1] != v.X.shape[1]:
            raise ValueError("Quadrature mismatch: trial and test functions "
                             "should have same number of integration points.")

        nt = u.nelems
        dx = u.dx
        w = FormDict({**u.default_parameters(), **self.dictify(kwargs)})

        # initialize COO data structures
        sz = u.Nbfun * v.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        # loop over the indices of local stiffness matrix
        for j in range(u.Nbfun):
            for i in range(v.Nbfun):
                ixs = slice(nt * (v.Nbfun * j + i),
                            nt * (v.Nbfun * j + i + 1))
                rows[ixs] = v.element_dofs[i]
                cols[ixs] = u.element_dofs[j]
                data[ixs] = self._kernel(u.basis[j], v.basis[i], w, dx)

        # TODO: allow user to change, e.g. cuda or petsc
        return self._assemble_scipy_matrix(data, rows, cols, (v.N, u.N))

    def _kernel(self, u, v, w, dx):
        return np.sum(self.form(*u, *v, w) * dx, axis=1)
