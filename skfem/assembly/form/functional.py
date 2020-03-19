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
                  w: Dict[str, DiscreteField] = {}) -> ndarray:
        w = FormDict({**v.default_parameters(), **self.dictify(w)})
        return self._kernel(w, v.dx)

    def assemble(self,
                 v: Basis,
                 w: Dict[str, DiscreteField] = {}) -> float:
        return np.sum(self.elemental(v, w))


def functional(form: Callable) -> Functional:

    # for backwards compatibility
    from .form_parameters import FormParameters

    # TODO: deprecate
    class ClassicFunctional(Functional):

        def _kernel(self, w, dx):
            W = {k: w[k].f for k in w}
            if 'w' in w:
                W['dw'] = w['w'].df
            return np.sum(self.form(w=FormParameters(**W)) * dx, axis=1)

    return ClassicFunctional(form)
