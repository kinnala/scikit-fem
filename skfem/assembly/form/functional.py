from typing import Dict, Optional

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
        if self.form is None:
            raise Exception("Form function handle not defined.")
        return np.sum(self.form(w) * dx, axis=1)

    def elemental(self,
                  v: Basis,
                  **kwargs) -> ndarray:
        w = FormDict({**v.default_parameters(), **self.dictify(kwargs)})
        return self._kernel(w, v.dx)

    def assemble(self,
                 u_basis: Basis,
                 v_basis: Optional[Basis] = None,
                 **kwargs) -> float:
        assert v_basis is None
        v_basis = u_basis
        return np.sum(self.elemental(v_basis, **kwargs))
