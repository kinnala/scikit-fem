from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray

from ...element import DiscreteField
from ..basis import Basis


class FormExtraParams(dict):
    """Passed to forms as 'w'."""

    def __getattr__(self, attr):
        return self[attr].value


class Form:

    form: Optional[Callable] = None

    def __init__(self,
                 form: Optional[Callable] = None,
                 dtype: type = np.float64,
                 nthreads: int = 0):
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

    def assemble(self,
                 ubasis: Basis,
                 vbasis: Optional[Basis] = None,
                 **kwargs) -> Any:
        raise NotImplementedError

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
