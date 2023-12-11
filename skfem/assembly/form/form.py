import logging
import warnings
import numbers
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional, Union, Type
from inspect import signature

import numpy as np
from numpy import ndarray

from ...element import DiscreteField
from .coo_data import COOData


logger = logging.getLogger(__name__)


class FormExtraParams(dict):
    """Passed to forms as 'w'."""

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError("Attribute '{}' not found in 'w'.".format(attr))


class Form:

    form: Optional[Callable] = None

    def __init__(self,
                 form: Optional[Union[Callable, 'Form']] = None,
                 dtype: Union[Type[np.float64],
                              Type[np.complex64]] = np.float64,
                 nthreads: int = 0,
                 **params):
        self.form = form.form if isinstance(form, Form) else form
        self.nargs = (len(signature(self.form).parameters)
                      if self.form is not None
                      else None)
        self.dtype = dtype
        self.nthreads = nthreads
        self.params = params

    def partial(self, *args, **kwargs):
        form = deepcopy(self)
        name = form.form.__name__
        form.form = partial(form.form, *args, **kwargs)
        form.form.__name__ = name
        return form

    def block(self, *args):
        form = deepcopy(self)
        name = form.form.__name__
        form.form = lambda *arg: self.form(
            *[arg[k] if args[k] == j else arg[k].zeros()
              for k in range(len(arg) - 1)
              for j in range(int((self.nargs - 1) / (len(arg) - 1)))],
            arg[-1]
        )
        form.form.__name__ = name
        return form

    def __call__(self, *args):
        if self.form is None:  # decorate
            return type(self)(form=args[0],
                              dtype=self.dtype,
                              nthreads=self.nthreads,
                              **self.params)
        return self.assemble(self.kernel(*args))

    def assemble(self, *args, **kwargs) -> Any:
        assert self.form is not None
        logger.info("Assembling '{}'.".format(self.form.__name__))
        out = (COOData(*self._assemble(*args, **kwargs))  # type: ignore
               .todefault())
        logger.info("Assembling finished.")
        return out

    def coo_data(self, *args, **kwargs) -> COOData:
        return COOData(*self._assemble(*args, **kwargs))  # type: ignore

    @staticmethod
    def _normalize_asm_kwargs(w, basis):
        """Support additional input formats for 'w'."""
        for k in w:
            if isinstance(w[k], DiscreteField):
                if w[k].shape[-1] != basis.X.shape[-1]:
                    raise ValueError("Quadrature mismatch: '{}' should have "
                                     "same number of integration points as "
                                     "the basis object.".format(k))
            elif isinstance(w[k], numbers.Number):
                # scalar parameter is passed to form
                continue
            elif isinstance(w[k], tuple):
                # asm() product index is of type tuple
                continue
            elif isinstance(w[k], ndarray) and len(w[k].shape) == 1:
                # interpolate DOF arrays at quadrature points
                w[k] = basis.interpolate(w[k])
            elif isinstance(w[k], ndarray) and len(w[k].shape) > 1:
                w[k] = DiscreteField(w[k])
            elif isinstance(w[k], list):
                warnings.warn("Use Basis.interpolate instead of passing lists "
                              "to assemble", DeprecationWarning)
                # for backwards-compatibility
                w[k] = DiscreteField(np.array([z.value for z in w[k]]),
                                     np.array([z.grad for z in w[k]]))
            else:
                raise ValueError("The given type '{}' for the list of extra "
                                 "form parameters w cannot be converted to "
                                 "DiscreteField.".format(type(w[k])))
        return w
