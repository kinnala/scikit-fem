from typing import Callable, Optional

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
                 dtype=np.float,
                 **kwargs) -> ndarray:

        assert v is None
        v = u

        nt = v.nelems
        dx = v.dx
        w = FormDict({**v.default_parameters(), **self.dictify(kwargs)})

        # initialize COO data structures
        sz = v.Nbfun * nt
        data = np.zeros(sz, dtype=dtype)
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


def linear_form(form: Callable) -> LinearForm:

    # for backwards compatibility
    from .form_parameters import FormParameters

    import warnings
    warnings.warn("The old style @linear_form wrapper is deprecated. "
                  "Consider using the new style forms, defined via "
                  "@LinearForm.", DeprecationWarning)

    class ClassicLinearForm(LinearForm):

        def _kernel(self, v, w, dx):
            v = v[0]
            W = {k: w[k].f for k in w}
            if 'w' in w:
                W['dw'] = w['w'].df
            if v.ddf is not None:
                return np.sum(self.form(v=v.f, dv=v.df, ddv=v.ddf,
                                        w=FormParameters(**W)) * dx, axis=1)
            else:
                return np.sum(self.form(v=v.f, dv=v.df,
                                        w=FormParameters(**W)) * dx, axis=1)

    return ClassicLinearForm(form)
