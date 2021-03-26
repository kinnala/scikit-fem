"""Helper functions for defining forms."""

from typing import Union
import numpy as np
from numpy import ndarray, zeros_like
from skfem.element import DiscreteField


FieldOrArray = Union[DiscreteField, ndarray]


def grad(u: DiscreteField):
    """Gradient."""
    return u.grad


def div(u: DiscreteField):
    """Divergence."""
    if u.div is not None:
        return u.div
    elif u.grad is not None:
        try:
            return np.einsum('ii...', u.grad)
        except ValueError:  # one-dimensional u?
            return u.grad[0]
    raise NotImplementedError


def curl(u: DiscreteField):
    """Curl."""
    if u.curl is not None:
        return u.curl
    elif u.grad is not None:
        if u.grad.shape[0] == 2:
            return np.array([u.grad[1], -u.grad[0]])
    raise NotImplementedError


def d(u: DiscreteField):
    """Gradient, divergence or curl."""
    if u.grad is not None:
        return u.grad
    elif u.div is not None:
        return u.div
    elif u.curl is not None:
        return u.curl
    raise NotImplementedError


def sym_grad(u: DiscreteField):
    """Symmetric gradient."""
    return .5 * (u.grad + transpose(u.grad))


def dd(u: DiscreteField):
    """Hessian (if available)."""
    return u.hess


def ddd(u):
    """Third derivative (if available)."""
    return u.grad3


def dddd(u):
    """Fourth derivative (if available)."""
    return u.grad4


def dot(u: FieldOrArray, v: FieldOrArray):
    """Dot product."""
    return np.einsum('i...,i...', u, v)


def ddot(u: FieldOrArray, v: FieldOrArray):
    """Double dot product."""
    return np.einsum('ij...,ij...', u, v)


def dddot(u: FieldOrArray, v: FieldOrArray):
    """Triple dot product."""
    return np.einsum('ijk...,ijk...', u, v)


def prod(u: FieldOrArray, v: FieldOrArray, w: FieldOrArray = None):
    """Tensor product."""
    if w is None:
        return np.einsum('i...,j...->ij...', u, v)
    return np.einsum('i...,j...,k...->ijk...', u, v, w)


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


def identity(w, N=None):
    """Create identity matrix."""
    if isinstance(w, DiscreteField):
        proto = w.value
    elif isinstance(w, ndarray):
        proto = w
    else:
        raise NotImplementedError

    if N is None:
        if len(proto.shape) > 2:
            N = proto.shape[-3]
        else:
            raise ValueError("Cannot deduce the size of the identity matrix. "
                             "Give an explicit keyword argument N.")

    return eye(np.ones(proto.shape[-2:]), N)


def det(A):
    """
    Determinant of an array `A` over trailing axis (if any).

    Parameters
    ----------
    A : (N, N,...) numpy.ndarray
        N = 2 or 3
        Input array whose determinant is to be computed

    Returns
    -------
    det : (...) numpy.ndarray
        Determinant of `A`.

    """
    detA = zeros_like(A[0, 0])
    if A.shape[0] == 3:
        detA = A[0, 0] * (A[1, 1] * A[2, 2] -
                          A[1, 2] * A[2, 1]) -\
               A[0, 1] * (A[1, 0] * A[2, 2] -
                          A[1, 2] * A[2, 0]) +\
               A[0, 2] * (A[1, 0] * A[2, 1] -
                          A[1, 1] * A[2, 0])
    elif A.shape[0] == 2:
        detA = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    return detA


def inv(A):
    """Inverse of an array `A` over trailing axis (if any)

    Parameters
    ----------
    A : (N, N,...) numpy.ndarray
        N = 2 or 3
        Input array whose inverse is to be computed

    Returns
    -------
    Ainv : (N, N,...) numpy.ndarray
        Inverse of `A`.

    """
    invA = zeros_like(A)
    detA = det(A)
    if A.shape[0] == 3:
        invA[0, 0] = (-A[1, 2] * A[2, 1] +
                      A[1, 1] * A[2, 2]) / detA
        invA[1, 0] = (A[1, 2] * A[2, 0] -
                      A[1, 0] * A[2, 2]) / detA
        invA[2, 0] = (-A[1, 1] * A[2, 0] +
                      A[1, 0] * A[2, 1]) / detA
        invA[0, 1] = (A[0, 2] * A[2, 1] -
                      A[0, 1] * A[2, 2]) / detA
        invA[1, 1] = (-A[0, 2] * A[2, 0] +
                      A[0, 0] * A[2, 2]) / detA
        invA[2, 1] = (A[0, 1] * A[2, 0] -
                      A[0, 0] * A[2, 1]) / detA
        invA[0, 2] = (-A[0, 2] * A[1, 1] +
                      A[0, 1] * A[1, 2]) / detA
        invA[1, 2] = (A[0, 2] * A[1, 0] -
                      A[0, 0] * A[1, 2]) / detA
        invA[2, 2] = (-A[0, 1] * A[1, 0] +
                      A[0, 0] * A[1, 1]) / detA
    elif A.shape[0] == 2:
        invA[0, 0] = A[1, 1] / detA
        invA[0, 1] = -A[0, 1] / detA
        invA[1, 0] = -A[1, 0] / detA
        invA[1, 1] = A[0, 0] / detA
    return invA
