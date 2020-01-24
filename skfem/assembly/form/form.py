from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix

from .form_parameters import FormParameters
from ..basis import Basis
from ...element import DiscreteField


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
    def parameters(w, u):
        if isinstance(w, DiscreteField):
            w = {'w': w}
        elif isinstance(w, ndarray):
            w = {'w': DiscreteField(w)}
        elif isinstance(w, list):
            w = {'w': DiscreteField(np.array([z.f for z in w]))}
        elif isinstance(w, tuple):
            w = {'w': DiscreteField(*w)}
        return {**w, **u.default_parameters()}

    @staticmethod
    def _assemble_scipy_matrix(data, rows, cols, shape=None):
        K = coo_matrix((data, (rows, cols)), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    @staticmethod
    def _assemble_numpy_vector(data, rows, cols, shape=None):
        return coo_matrix((data, (rows, cols)),
                          shape=shape).toarray().T[0]
