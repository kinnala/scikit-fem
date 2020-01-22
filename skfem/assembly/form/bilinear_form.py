from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from .form import Form
from .form_parameters import FormParameters
from ..basis import Basis


class BilinearForm(Form):

    def assemble(self,
                 u: Basis,
                 v: Optional[Basis] = None,
                 w: Optional[Any] = (None, None, None)) -> csr_matrix:
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
        else:
            assert u.intorder == v.intorder, "Quadrature mismatch"

        nt = u.nelems
        dx = u.dx
        w = self.parameters(w, u)

        # initialize COO data structures
        sz = u.Nbfun * v.Nbfun * nt
        data = np.zeros(sz)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        # create sparse matrix indexing
        for j in range(u.Nbfun):
            for i in range(v.Nbfun):
                # find correct location in data, rows, cols
                ixs = slice(nt * (v.Nbfun * j + i),
                            nt * (v.Nbfun * j + i + 1))
                rows[ixs] = v.element_dofs[i]
                cols[ixs] = u.element_dofs[j]
                data[ixs] = np.sum(self.form(*u.basis[j], *v.basis[i], w) * dx, axis=1)

        K = coo_matrix((data, (rows, cols)), shape=(v.N, u.N))
        K.eliminate_zeros()
        return K.tocsr()


def bilinear_form(form: Callable) -> BilinearForm:
    return BilinearForm(form)
