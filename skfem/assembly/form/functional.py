from typing import Any, Callable, Optional

from numpy import ndarray
import numpy as np

from .form import Form
from .form_parameters import FormParameters
from ..global_basis import GlobalBasis


class Functional(Form):

    def kernel(self,
               w: FormParameters,
               dx: ndarray) -> ndarray:
        return np.sum(self.form(w) * dx, axis=1)

    def elemental(self,
                  vbasis: GlobalBasis,
                  w: Optional[Any] = (None, None, None)) -> ndarray:
        return self.kernel(self.parameters(w, vbasis), vbasis.dx)

    def assemble(self,
                 vbasis: GlobalBasis,
                 w: Optional[Any] = (None, None, None)) -> float:
        return sum(self.elemental(vbasis, w))


def functional(form: Callable) -> Functional:
    return Functional(form)
