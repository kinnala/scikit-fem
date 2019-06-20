from typing import Any, Callable, Optional

from numpy import ndarray
import numpy as np

from .form import Form
from .form_parameters import FormParameters
from ..global_basis import GlobalBasis


class Functional(Form):

    def kernel(self,
               w: FormParameters,
               dx: ndarray) -> None:
        return np.sum(self.form(w) * dx, axis=1)

    def assemble(self,
                 ubasis: GlobalBasis,
                 vbasis: Optional[GlobalBasis] = None,
                 w: Optional[Any] = (None, None, None),
                 nthreads: Optional[int] = 1) -> ndarray:
        assert vbasis is None
        vbasis = ubasis
        dx = vbasis.dx

        if type(w) is list:
            w = zip(*w)
        elif type(w) is ndarray:
            w = (w, None, None)

        w = FormParameters(*w, **vbasis.default_parameters())

        return self.kernel(w, dx)


def functional(form: Callable) -> Functional:
    return Functional(form)
