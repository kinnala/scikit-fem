import warnings
from typing import Any, Callable, Optional, Tuple, Union

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix

from .form_parameters import FormParameters
from ..basis import Basis
from ...element import DiscreteField


class FormDict(dict):
    """Passed to forms as 'w'."""

    def __getattr__(self, attr):
        return self[attr].value


class Form:

    def __init__(self, form: Callable):
        self.form = form

    def __call__(self, *args):
        return self.assemble(self.kernel(*args))

    def _kernel(self):
        raise NotImplementedError

    def assemble(self):
        raise NotImplementedError

    @staticmethod
    def dictify(w):
        """Support some legacy input formats for 'w'."""
        if not isinstance(w, dict):

            warnings.warn(("Previous versions of the library "
                           "supported multiple formats "
                           "for the prespecified field w. "
                           "In future, we support only "
                           "Dict[str, DiscreteField] which "
                           "can contain also multiple fields. "
                           "In most cases, you should be able to "
                           "simply replace w with {'w': w} to suppress "
                           "this warning. "), DeprecationWarning)

            if isinstance(w, DiscreteField):
                w = {'w': w}
            elif isinstance(w, ndarray):
                w = {'w': DiscreteField(w)}
            elif isinstance(w, list):
                w = {'w': DiscreteField(np.array([z['w'].f for z in w]),
                                        np.array([z['w'].df for z in w]))}
            elif isinstance(w, tuple):
                w = {'w': DiscreteField(*w)}
            else:
                raise ValueError("The given type '{}' for the list of extra "
                                 "form parameters w cannot be converted to "
                                 "Dict[str, DiscreteField].".format(type(w)))
        return w

    @staticmethod
    def _assemble_scipy_matrix(data, rows, cols, shape=None):
        K = coo_matrix((data, (rows, cols)), shape=shape)
        K.eliminate_zeros()
        return K.tocsr()

    @staticmethod
    def _assemble_numpy_vector(data, rows, cols, shape=None):
        return coo_matrix((data, (rows, cols)),
                          shape=shape).toarray().T[0]
