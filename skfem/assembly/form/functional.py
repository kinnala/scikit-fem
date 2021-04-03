from typing import Dict, Optional, Any

from numpy import ndarray

from .form import Form, FormExtraParams
from ..basis import Basis
from ...element import DiscreteField


class Functional(Form):
    """A functional for postprocessing finite element solution.

    Used similarly as :class:`~skfem.assembly.BilinearForm` with the expection
    that forms take one parameter ``w``.

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
        """Evaluate the functional elementwise."""
        w = FormExtraParams({
            **v.default_parameters(),
            **self.dictify(kwargs),
        })
        return self._kernel(w, v.dx)

    def assemble(self,
                 ubasis: Basis,
                 vbasis: Optional[Basis] = None,
                 **kwargs) -> Any:
        """Evaluate the functional to a scalar.

        Parameters
        ----------
        ubasis
            The :class:`~skfem.assembly.Basis` for filling the default
            parameters of ``w`` and integrating over the domain.
        **kwargs
            Any additional keyword arguments are appended to ``w``.

        """
        assert vbasis is None
        vbasis = ubasis
        return self.elemental(vbasis, **kwargs).sum(-1)
