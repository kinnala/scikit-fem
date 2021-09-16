from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray

from ...element import DiscreteField
from .coo_data import COOData


class FormExtraParams(dict):
    """Passed to forms as 'w'."""

    def __getattr__(self, attr):
        if attr in self:
            if hasattr(self[attr], 'value'):
                return self[attr].value
            return self[attr]
        raise ValueError


class Form:

    form: Optional[Callable] = None

    def __init__(self,
                 form: Optional[Callable] = None,
                 dtype: type = np.float64,
                 nthreads: int = 0,
                 inverse: bool = False):
        self.form = form.form if isinstance(form, Form) else form
        self.dtype = dtype
        self.nthreads = nthreads

    def partial(self, *args, **kwargs):
        form = deepcopy(self)
        form.form = partial(form.form, *args, **kwargs)
        return form

    def __call__(self, *args):
        if self.form is None:  # decorate
            return type(self)(form=args[0],
                              dtype=self.dtype,
                              nthreads=self.nthreads)
        return self.assemble(self.kernel(*args))

    def assemble(self, *args, **kwargs) -> Any:
        return (COOData(*self._assemble(*args, **kwargs))  # type: ignore
                .todefault())

    def coo_data(self, *args, **kwargs) -> COOData:
        return COOData(*self._assemble(*args, **kwargs))  # type: ignore

    @staticmethod
    def dictify(w, basis):
        """Support additional input formats for 'w'."""
        for k in w:
            if isinstance(w[k], DiscreteField):
                continue
            elif isinstance(w[k], ndarray) and len(w[k].shape) == 2:
                w[k] = DiscreteField(w[k])
            elif isinstance(w[k], ndarray) and len(w[k].shape) == 1:
                w[k] = basis.interpolate(w[k])
            else:
                raise ValueError("The given type '{}' for the list of extra "
                                 "form parameters w cannot be converted to "
                                 "DiscreteField.".format(type(w[k])))
        return w
