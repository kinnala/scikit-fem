from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .form import Form, FormExtraParams
from ..basis import AbstractBasis


class LinearForm(Form):
    """A linear form for finite element assembly.

    Used similarly as :class:`~skfem.assembly.BilinearForm` with the expection
    that forms take two parameters ``v`` and ``w``.

    """

    def _assemble(self,
                  ubasis: AbstractBasis,
                  vbasis: Optional[AbstractBasis] = None,
                  **kwargs) -> Tuple[ndarray,
                                     ndarray,
                                     Tuple[int],
                                     Tuple[int]]:

        assert vbasis is None
        vbasis = ubasis

        nt = vbasis.nelems
        dx = vbasis.dx
        w = FormExtraParams({
            **vbasis.default_parameters(),
            **self.dictify(kwargs, ubasis),
            'sign': ubasis._sign,
        })

        # initialize COO data structures
        sz = vbasis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int64)

        for i in range(vbasis.Nbfun):
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.element_dofs[i]
            data[ixs] = self._kernel(vbasis.basis[i], w, dx)

        return np.array([rows]), data, (vbasis.N,), (vbasis.Nbfun,)

    def _kernel(self, v, w, dx):
        return np.sum(self.form(*v, w) * dx, axis=1)
