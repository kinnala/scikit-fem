import logging
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray

from ...element import DiscreteField
from .coo_data import COOData


logger = logging.getLogger(__name__)


class FormExtraParams(dict):
    """Passed to forms as 'w'."""

    def __getattr__(self, attr):
        if attr[:4] == 'sign':
            # for backwards compatibility
            ix = int(attr[4]) - 1 if len(attr) == 5 else 0
            if hasattr(self, 'idx') and ix < len(self['idx']):
                return (-1.) ** self['idx'][ix]
            return 1.
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
        name = form.form.__name__
        form.form = partial(form.form, *args, **kwargs)
        form.form.__name__ = name
        return form

    def __call__(self, *args):
        if self.form is None:  # decorate
            return type(self)(form=args[0],
                              dtype=self.dtype,
                              nthreads=self.nthreads)
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
    def dictify(w, basis):
        """Support additional input formats for 'w'."""
        for k in w:
            if isinstance(w[k], ndarray) and len(w[k].shape) == 2:
                w[k] = DiscreteField(w[k])
            elif isinstance(w[k], ndarray) and len(w[k].shape) == 1:
                w[k] = basis.interpolate(w[k])
        return w
