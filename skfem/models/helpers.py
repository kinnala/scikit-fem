"""Helper functions for defining forms."""

from typing import Union
import numpy as np
from numpy import ndarray
from skfem.element import DiscreteField


FieldOrArray = Union[DiscreteField, ndarray]


def grad(u: DiscreteField):
    """Gradient."""
    return u[1]


def sym_grad(u: DiscreteField):
    """Symmetric gradient."""
    du = grad(u)
    return .5 * (du + transpose(du))


def div(u: DiscreteField):
    """Divergence."""
    return np.einsum('ii...', grad(u))


def dot(u: FieldOrArray, v: FieldOrArray):
    """Dot product."""
    u = u.f if isinstance(u, DiscreteField) else u
    v = v.f if isinstance(v, DiscreteField) else v
    return np.einsum('i...,i...', u, v)


def ddot(u: FieldOrArray, v: FieldOrArray):
    """Double dot product."""
    u = u.f if isinstance(u, DiscreteField) else u
    v = v.f if isinstance(v, DiscreteField) else v
    return np.einsum('ij...,ij...', u, v)


def prod(u: FieldOrArray, v: FieldOrArray):
    """Tensor product."""
    u = u.f if isinstance(u, DiscreteField) else u
    v = v.f if isinstance(v, DiscreteField) else v
    return np.einsum('i...,j...->ij...', u, v)


def trace(T):
    """Trace of matrix."""
    return np.einsum('ii...', T)


def transpose(T):
    """Transpose of matrix."""
    return np.einsum('ij...->ji...', T)


def eye(w, n):
    """Create diagonal matrix with w on diagonal."""
    return np.array([[w if i == j else 0. * w for i in range(n)]
                     for j in range(n)])
