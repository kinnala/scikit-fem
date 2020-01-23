from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray

from .form import Form
from .form_parameters import FormParameters
from ..basis import Basis


class BilinearForm(Form):

    def assemble(self,
                 u: Basis,
                 v: Optional[Basis] = None,
                 w: Optional[Any] = (None, None, None)) -> Any:
        """Return sparse CSR matrix discretizing form

        Parameters
        ----------
        w
            A tuple of ndarrays. In the form definition:

              * :code:`w[0]` is accessible as :code:`w.w`,
              * :code:`w[1]` is accessible as :code:`w.dw`, and
              * :code:`w[2]` is accessible as :code:`w.ddw`.

            The output of :meth:`~skfem.assembly.Basis.interpolate`
            can be passed directly to this parameter.

        """
        if v is None:
            v = u
        elif u.intorder != v.intorder:
            raise ValueError("Quadrature mismatch: trial and test functions "
                             "should have same number of integration points.")

        nt = u.nelems
        dx = u.dx
        w = self.parameters(w, u)

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
                data[ixs] = self._eval_local_matrices(u.basis[j], v.basis[i], w, dx)

        # TODO: allow user to change, e.g. cuda or petsc
        return self._assemble_scipy_matrix(data, rows, cols, (v.N, u.N))

    def _eval_local_matrices(self, u, v, w, dx):
        return np.sum(self.form(u, v, w) * dx, axis=1)


def bilinear_form(form: Callable) -> BilinearForm:

    # for backwards compatibility
    def eval_form(self, u, v, w, dx):
        if u.ddf is not None:
            return np.sum(self.form(u=u.f, du=u.df, ddu=u.ddf,
                                    v=v.f, dv=v.df, ddv=v.ddf,
                                    w=w) * dx, axis=1)
        else:
            return np.sum(self.form(u=u.f, du=u.df,
                                    v=v.f, dv=v.df,
                                    w=w) * dx, axis=1)

    BilinearForm._eval_local_matrices = eval_form

    return BilinearForm(form)
