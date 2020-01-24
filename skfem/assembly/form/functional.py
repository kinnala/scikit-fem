from typing import Any, Callable, Optional

from numpy import ndarray
import numpy as np

from .form import Form
from .form_parameters import FormParameters
from ..basis import Basis


class Functional(Form):

    def _kernel(self,
                w: FormParameters,
                dx: ndarray) -> ndarray:
        return np.sum(self.form(w) * dx, axis=1)

    def elemental(self,
                  v: Basis,
                  w: Optional[Any] = {}) -> ndarray:
        return self._kernel(self.parameters(w, v), v.dx)

    def assemble(self,
                 v: Basis,
                 w: Optional[Any] = {}) -> float:
        return sum(self.elemental(v, w))


def functional(form: Callable) -> Functional:

    # for backwards compatibility
    def kernel(self, w, dx):
        W = {k: w[k].f for k in w}
        return np.sum(self.form(w=FormParameters(**W)) * dx, axis=1)

    Functional._kernel = kernel

    return Functional(form)
