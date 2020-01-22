from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix

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

        data = np.zeros(v.Nbfun * nt)
        rows = np.zeros(v.Nbfun * nt)
        cols = np.zeros(v.Nbfun * nt)

        for i in range(v.Nbfun):
            # find correct location in data,rows,cols
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = v.element_dofs[i]
            cols[ixs] = np.zeros(nt)
            data[ixs] = np.sum(self.form(*v.basis[i], w) * dx, axis=1)

        return coo_matrix((data, (rows, cols)),
                          shape=(v.N, 1)).toarray().T[0]


def linear_form(form: Callable) -> LinearForm:
    return LinearForm(form)
