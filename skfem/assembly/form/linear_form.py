from typing import Optional, Tuple

import numpy as np
from numpy import ndarray

from .form import Form, FormExtraParams
from ..basis import Basis
from .coo_data import COOData


class LinearForm(Form):
    """A linear form for finite element assembly.

    Used similarly as :class:`~skfem.assembly.BilinearForm` with the expection
    that forms take two parameters ``v`` and ``w``.

    """

    def _assemble(self,
                  ubasis: Basis,
                  vbasis: Optional[Basis] = None,
                  **kwargs) -> Tuple[ndarray,
                                     ndarray,
                                     ndarray,
                                     Tuple[int, int]]:

        assert vbasis is None
        vbasis = ubasis

        nt = vbasis.nelems
        dx = vbasis.dx
        w = FormExtraParams({
            **vbasis.default_parameters(),
            **self.dictify(kwargs),
        })

        # initialize COO data structures
        sz = vbasis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int64)
        cols = np.zeros(sz, dtype=np.int64)

        for i in range(vbasis.Nbfun):
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.element_dofs[i]
            cols[ixs] = np.zeros(nt)
            data[ixs] = self._kernel(vbasis.basis[i], w, dx)

        return data, rows, cols, (vbasis.N, 1)

    def coo_data(self, *args, **kwargs) -> COOData:
        return COOData(*self._assemble(*args, **kwargs))

    def assemble(self, *args, **kwargs) -> ndarray:
        """Assemble the linear form into a vector.

        Parameters
        ----------
        ubasis
            The :class:`~skfem.assembly.Basis` for ``v``.
        **kwargs
            Any additional keyword arguments are appended to ``w``.

        """
        return COOData(*self._assemble(*args, **kwargs)).toarray().T[0]

    def _kernel(self, v, w, dx):
        return np.sum(self.form(*v, w) * dx, axis=1)
