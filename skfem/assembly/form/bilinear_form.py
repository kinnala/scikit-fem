from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from .form import Form, BasisTuple
from .form_parameters import FormParameters
from ..basis import Basis


class BilinearForm(Form):

    def assemble(self,
                 ubasis: Basis,
                 vbasis: Optional[Basis] = None,
                 w: Optional[Any] = (None, None, None),
                 nthreads: Optional[int] = 1) -> csr_matrix:
        """return sparse CSR matrix discretizing form

        :param w: A tuple of ndarrays. In the form definition:

          * :code:`w[0]` is accessible as :code:`w.w`,

          * :code:`w[1]` is accessible as :code:`w.dw`, and

          * :code:`w[2]` is accessible as :code:`w.ddw`.

        The output of :meth:`~skfem.assembly.Basis.interpolate`
        can be passed directly to this parameter.

        """

        import threading
        from itertools import product

        if vbasis is None:
            vbasis = ubasis
        else:
            assert ubasis.intorder == vbasis.intorder, "Quadrature mismatch"

        nt = ubasis.nelems
        dx = ubasis.dx
        w = self.parameters(w, ubasis)

        # initialize COO data structures
        data = np.zeros((vbasis.Nbfun, ubasis.Nbfun, nt))
        rows = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)
        cols = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)

        # create sparse matrix indexing
        for j in range(ubasis.Nbfun):
            for i in range(vbasis.Nbfun):
                # find correct location in data, rows, cols
                ixs = slice(nt * (vbasis.Nbfun * j + i),
                            nt * (vbasis.Nbfun * j + i + 1))
                rows[ixs] = vbasis.element_dofs[i]
                cols[ixs] = ubasis.element_dofs[j]
                data[ixs] = np.sum(self.form(*ubasis[j], *vbasis[i], w) * dx,
                                   axis=1)

        K = coo_matrix((np.transpose(data, (1, 0, 2)).flatten('C'),
                        (rows, cols)),
                       shape=(vbasis.N, ubasis.N))
        K.eliminate_zeros()
        return K.tocsr()


def bilinear_form(form: Callable) -> BilinearForm:
    return BilinearForm(form)
