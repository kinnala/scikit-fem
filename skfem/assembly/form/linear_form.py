from typing import Optional
from functools import lru_cache

import numpy as np
from numpy import ndarray

from .form import Form, FormDict
from ..basis import Basis
from skfem.generic_utils import HashableNdArray


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
        dx = HashableNdArray(ubasis.dx)
        w = FormDict({**vbasis.default_parameters(), **self.dictify(kwargs)})

        # initialize COO data structures
        sz = vbasis.Nbfun * nt
        data = np.zeros(sz, dtype=self.dtype)
        rows = np.zeros(sz)
        cols = np.zeros(sz)

        ixs = 0  # Track index in the (data, rows, cols) triplets.
        for i in range(vbasis.Nbfun):
            d = self._kernel(vbasis.basis[i], w, dx)
            if (d != np.zeros_like(d)).any():
                r = vbasis.element_dofs[i]
                ix_slice = slice(ixs, ixs + len(r))
                rows[ix_slice] = r
                cols[ix_slice] = np.zeros(nt)
                data[ix_slice] = d
                ixs += len(r)

        return self._assemble_numpy_vector(
            data[0:ixs],
            rows[0:ixs],
            cols[0:ixs],
            (vbasis.N, 1)
        )

    @lru_cache(maxsize=128)
    def _kernel(self, v, w, dx):
        return np.sum(self.form(*v, w) * dx, axis=1)
