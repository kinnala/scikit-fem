from typing import Any, Callable, Optional, Dict

import numpy as np
from numpy import ndarray

from .form import Form, FormDict
from ..basis import Basis
from ...element import DiscreteField


class BilinearForm(Form):

    def assemble(self,
                 u: Basis,
                 v: Optional[Basis] = None,
                 w: Dict[str, DiscreteField] = {}) -> Any:

        if v is None:
            v = u
        elif u.intorder != v.intorder:
            raise ValueError("Quadrature mismatch: trial and test functions "
                             "should have same number of integration points.")

        nt = u.nelems
        dx = u.dx
        w = FormDict({**u.default_parameters(), **self.dictify(w)})

        # initialize COO data structures
        sz = u.Nbfun * v.Nbfun * nt
        data = np.zeros(sz)
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


def bilinear_form(form: Callable) -> BilinearForm:

    # for backwards compatibility
    from .form_parameters import FormParameters

    # TODO: deprecate
    class ClassicBilinearForm(BilinearForm):

        def _kernel(self, u, v, w, dx):
            u = u[0]
            v = v[0]
            W = {k: w[k].f for k in w}
            if 'w' in w:
                W['dw'] = w['w'].df
            if u.ddf is not None:
                return np.sum(self.form(u=u.f, du=u.df, ddu=u.ddf,
                                        v=v.f, dv=v.df, ddv=v.ddf,
                                        w=FormParameters(**W)) * dx, axis=1)
            else:
                return np.sum(self.form(u=u.f, du=u.df,
                                        v=v.f, dv=v.df,
                                        w=FormParameters(**W)) * dx, axis=1)

    return ClassicBilinearForm(form)
