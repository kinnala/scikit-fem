from typing import Dict

from numpy import ndarray
import numpy as np

from .form import Form, FormDict
from ..basis import Basis
from ...element import DiscreteField


class Functional(Form):
    """A functional for finite element assembly.

    Used similarly as :class:`~skfem.assembly.BilinearForm` with the expection
    that forms take one parameter `w`.

    """

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
