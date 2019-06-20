from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix

from .form import Form
from .form_parameters import FormParameters
from ..global_basis import GlobalBasis


class LinearForm(Form):

    def kernel(self,
               b: ndarray,
               ix: ndarray,
               vbasis: GlobalBasis,
               w: FormParameters,
               dx: ndarray) -> None:
        for i in ix:
            b[i] = np.sum(self.form(*vbasis[i], w) * dx, axis=1)

    def assemble(self,
                 ubasis: GlobalBasis,
                 vbasis: Optional[GlobalBasis] = None,
                 w: Optional[Any] = (None, None, None),
                 nthreads: Optional[int] = 1) -> ndarray:

        import threading

        assert vbasis is None
        vbasis = ubasis

        nt = vbasis.nelems
        dx = vbasis.dx
        w = self.parameters(w, vbasis)

        data = np.zeros((vbasis.Nbfun, nt))
        rows = np.zeros(vbasis.Nbfun * nt)
        cols = np.zeros(vbasis.Nbfun * nt)

        for i in range(vbasis.Nbfun):
            # find correct location in data,rows,cols
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.element_dofs[i, :]
            cols[ixs] = np.zeros(nt)

        indices = range(vbasis.Nbfun)

        threads = [threading.Thread(
            target=self.kernel,
            args=(data, ix, vbasis.basis, w, dx))
                   for ix in np.array_split(indices, nthreads, axis=0)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return coo_matrix((data.flatten('C'), (rows, cols)),
                          shape=(vbasis.N, 1)).toarray().T[0]


def linear_form(form: Callable) -> LinearForm:
    return LinearForm(form)
