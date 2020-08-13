from typing import Optional

import numpy as np
from numpy import ndarray

from .form import Form, FormDict
from ..basis import Basis


class LinearForm(Form):
    """A linear form for finite element assembly.

    Used similarly as :class:`~skfem.assembly.BilinearForm` with the expection
    that forms take two parameters `v` and `w`.

    """

    def assemble(self,
                 u: Basis,
                 v: Optional[Basis] = None,
                 **kwargs) -> ndarray:

        assert v is None
        v = u

        nt = v.nelems
        dx = v.dx
        w = FormDict({**v.default_parameters(), **self.dictify(kwargs)})

        # initialize COO data structures
        sz = v.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        for i in range(v.Nbfun):
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = v.element_dofs[i]
            cols[ixs] = np.zeros(nt)
            data[ixs] = self._kernel(v.basis[i], w, dx)

        return self._assemble_numpy_vector(data, rows, cols, (v.N, 1))

    def _kernel(self, v, w, dx):
        return np.sum(self.form(*v, w) * dx, axis=1)
