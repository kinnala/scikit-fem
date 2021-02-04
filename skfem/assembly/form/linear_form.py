from typing import Optional

import numpy as np
from numpy import ndarray

from .form import Form, FormDict
from ..basis import Basis


class LinearForm(Form):
    """A linear form for finite element assembly.

    Used similarly as :class:`~skfem.assembly.BilinearForm` with the expection
    that forms take two parameters ``v`` and ``w``.

    """

    def assemble(self,
                 ubasis: Basis,
                 vbasis: Optional[Basis] = None,
                 **kwargs) -> ndarray:
        """Assemble the linear form into a vector.

        Parameters
        ----------
        ubasis
            The :class:`~skfem.assembly.Basis` for ``v``.
        **kwargs
            Any additional keyword arguments are appended to ``w``.

        """

        assert vbasis is None
        vbasis = ubasis

        nt = vbasis.nelems
        dx = vbasis.dx
        w = FormDict({**vbasis.default_parameters(), **self.dictify(kwargs)})

        # initialize COO data structures
        sz = vbasis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        for i in range(vbasis.Nbfun):
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.element_dofs[i]
            cols[ixs] = np.zeros(nt)
            data[ixs] = self._kernel(vbasis.basis[i], w, dx)

        return self._assemble_numpy_vector(data, rows, cols, (vbasis.N, 1))

    def _kernel(self, v, w, dx):
        return np.sum(self.form(*v, w) * dx, axis=1)
