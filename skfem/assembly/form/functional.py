from typing import Any, Callable, Optional, Dict

from numpy import ndarray
import numpy as np

from .form import Form, FormDict
from ..basis import Basis
from ...element import DiscreteField


class Functional(Form):

    def _kernel(self,
                w: Dict[str, DiscreteField],
                dx: ndarray) -> ndarray:
        return np.sum(self.form(w) * dx, axis=1)

    def elemental(self,
                  v: Basis,
                  **kwargs) -> ndarray:
        w = FormDict({**v.default_parameters(), **self.dictify(kwargs)})
        return self._kernel(w, v.dx)

    def assemble(self,
                 v: Basis,
                 **kwargs) -> float:
        return np.sum(self.elemental(v, **kwargs))


def functional(form: Callable) -> Functional:

    # for backwards compatibility
    from .form_parameters import FormParameters

    import warnings
    warnings.warn("The old style @functional wrapper is deprecated. "
                  "Consider using the new style forms, defined via "
                  "@Functional.", DeprecationWarning)

    class ClassicFunctional(Functional):

        def _kernel(self, w, dx):
            W = {k: w[k].f for k in w}
            if 'w' in w:
                W['dw'] = w['w'].df
            return np.sum(self.form(w=FormParameters(**W)) * dx, axis=1)

    return ClassicFunctional(form)
