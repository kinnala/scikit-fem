from typing import Any, Callable, Optional

from numpy import ndarray

from .form_parameters import FormParameters
from ..global_basis import GlobalBasis


class Form:

    def __init__(self, form: Callable):
        self.form = form

    def __call__(self, *args):
        return self.assemble(self.kernel(*args))

    def kernel(self):
        raise NotImplementedError

    def assemble(self):
        raise NotImplementedError

    @staticmethod
    def parameters(w: Optional[Any], ubasis: GlobalBasis) -> FormParameters:
        if type(w) is list:
            w = zip(*w)
        elif type(w) is ndarray:
            w = (w, None, None)
        return FormParameters(*w, **ubasis.default_parameters())
