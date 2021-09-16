from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .form import Form, FormExtraParams
from ..basis import AbstractBasis


class TrilinearForm(Form):

    def _assemble(self,
                  ubasis: AbstractBasis,
                  vbasis: Optional[AbstractBasis] = None,
                  wbasis: Optional[AbstractBasis] = None,
                  **kwargs) -> Tuple[ndarray,
                                     ndarray,
                                     Tuple[int, int, int],
                                     Tuple[int, int, int]]:

        if vbasis is None:
            vbasis = ubasis
        if wbasis is None:
            wbasis = ubasis

        nt = ubasis.nelems
        dx = ubasis.dx
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self.dictify(kwargs, ubasis),
            'sign': ubasis._sign,
            'sign1': ubasis._sign,
            'sign2': vbasis._sign,
            'sign3': wbasis._sign,
        })

        # initialize COO data structures
        sz = (ubasis.Nbfun, vbasis.Nbfun, wbasis.Nbfun, nt)
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int64)
        cols = np.zeros(sz, dtype=np.int64)
        mats = np.zeros(sz, dtype=np.int64)

        # loop over the indices of local stiffness matrix
        for k in range(ubasis.Nbfun):
            for j in range(vbasis.Nbfun):
                for i in range(wbasis.Nbfun):
                    mats[k, j, i] = wbasis.element_dofs[i]
                    rows[k, j, i] = vbasis.element_dofs[j]
                    cols[k, j, i] = ubasis.element_dofs[k]
                    data[k, j, i] = self._kernel(
                        ubasis.basis[k],
                        vbasis.basis[j],
                        wbasis.basis[i],
                        wdict,
                        dx,
                    )

        return (
            np.array([
                mats.flatten(),
                rows.flatten(),
                cols.flatten(),
            ]),
            data.flatten(),
            (wbasis.N, vbasis.N, ubasis.N),
            (ubasis.Nbfun, vbasis.Nbfun, wbasis.Nbfun),
        )

    def _kernel(self, u, v, w, params, dx):
        return np.sum(self.form(*u, *v, *w, params) * dx, axis=1)
