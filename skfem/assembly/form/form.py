from typing import Any, Callable, Optional, Tuple, Union

from numpy import ndarray
from scipy.sparse import coo_matrix

from .form_parameters import FormParameters
from ..basis import Basis


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
    def parameters(w: Optional[Any], u: Basis) -> FormParameters:
        if type(w) is list:
            w = zip(*w)
        elif type(w) is ndarray:
            w = (w, None, None)
        return FormParameters(*w, **u.default_parameters())

    @staticmethod
    def _assemble_scipy_matrix(data, rows, cols, shape=None):
        K = coo_matrix((data, (rows, cols)), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    @staticmethod
    def _assemble_numpy_vector(data, rows, cols, shape=None):
        return coo_matrix((data, (rows, cols)),
                          shape=shape).toarray().T[0]
