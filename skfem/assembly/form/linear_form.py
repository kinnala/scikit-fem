from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray

from .form import Form
from .form_parameters import FormParameters
from ..basis import Basis


class LinearForm(Form):

    def assemble(self,
                 u: Basis,
                 v: Optional[Basis] = None,
                 w: Optional[Any] = (None, None, None)) -> ndarray:

        assert v is None
        v = u

        nt = v.nelems
        dx = v.dx
        w = self.parameters(w, v)

        # initialize COO data structures
        sz = v.Nbfun * nt
        data = np.zeros(sz)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        for i in range(v.Nbfun):
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = v.element_dofs[i]
            cols[ixs] = np.zeros(nt)
            data[ixs] = self._eval_local_vectors(v.basis[i], w, dx)

        return self._assemble_numpy_vector(data, rows, cols, (v.N, 1))

    def _eval_local_vectors(self, v, w, dx):
        return np.sum(self.form(v, w) * dx, axis=1)


def linear_form(form: Callable) -> LinearForm:

    # for backwards compatibility
    def eval_form(self, v, w, dx):
        if v.ddf is not None:
            return np.sum(self.form(v=v.f, dv=v.df, ddv=v.ddf,
                                    w=w) * dx, axis=1)
        else:
            return np.sum(self.form(v=v.f, dv=v.df,
                                    w=w) * dx, axis=1)

    LinearForm._eval_local_vectors = eval_form

    return LinearForm(form)
