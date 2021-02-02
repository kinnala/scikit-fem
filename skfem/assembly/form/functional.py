from typing import Dict, Optional, Any
from numbers import Number

from numpy import ndarray

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
        return (self.form(w) * dx).sum(-1)

    def elemental(self,
                  v: Basis,
                  **kwargs) -> ndarray:
        w = FormDict({**v.default_parameters(), **self.dictify(kwargs)})
        return self._kernel(w, v.dx)

    def assemble(self,
                 ubasis: Basis,
                 vbasis: Optional[Basis] = None,
                 **kwargs) -> Any:
        assert vbasis is None
        vbasis = ubasis
        return self.elemental(vbasis, **kwargs).sum(-1)
