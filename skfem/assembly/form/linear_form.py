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
                 u_basis: Basis,
                 v_basis: Optional[Basis] = None,
                 **kwargs) -> ndarray:

        assert v_basis is None
        v_basis = u_basis

        nt = v_basis.nelems
        dx = v_basis.dx
        w = FormDict({**v_basis.default_parameters(), **self.dictify(kwargs)})

        # initialize COO data structures
        sz = v_basis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        for i in range(v_basis.Nbfun):
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = v_basis.element_dofs[i]
            cols[ixs] = np.zeros(nt)
            data[ixs] = self._kernel(v_basis.basis[i], w, dx)

        return self._assemble_numpy_vector(data, rows, cols, (v_basis.N, 1))

    def _kernel(self, v, w, dx):
        return np.sum(self.form(*v, w) * dx, axis=1)
