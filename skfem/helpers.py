"""Helper functions for defining forms."""

from typing import Union, Optional
import numpy as np
from numpy import ndarray, zeros_like
from skfem.element import DiscreteField
from skfem.assembly.form.form import FormExtraParams


FieldOrArray = Union[DiscreteField, ndarray]


def jump(w: FormExtraParams, *args):
    if not hasattr(w, 'idx'):
        raise NotImplementedError("jump() can be used only if the form is "
                                  "assembled through asm().")
    out = []
    for i, arg in enumerate(args):
        out.append((-1.) ** w.idx[i] * arg)
    return out[0] if len(out) == 1 else tuple(out)


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
    """Hessian (for :class:`~skfem.element.ElementGlobal`)."""
    return u.hess


def ddd(u: DiscreteField):
    """Third derivative (for :class:`~skfem.element.ElementGlobal`)."""
    return u.grad3


def dddd(u: DiscreteField):
    """Fourth derivative (for :class:`~skfem.element.ElementGlobal`)."""
    return u.grad4


def inner(u: FieldOrArray, v: FieldOrArray):
    """Inner product between any matching tensors."""
    if isinstance(u, tuple) and isinstance(v, tuple):
        # support for ElementComposite
        out = []
        for i in range(len(v)):
            out.append(inner(u[i], v[i]))
        return sum(out)
    if len(u.shape) == 2:
        return u * v
    elif len(u.shape) == 3:
        return dot(u, v)
    elif len(u.shape) == 4:
        return ddot(u, v)
    raise NotImplementedError


def dot(u: FieldOrArray, v: FieldOrArray):
    """Dot product."""
    return np.einsum('i...,i...', u, v)


def ddot(u: FieldOrArray, v: FieldOrArray):
    """Double dot product."""
    return np.einsum('ij...,ij...', u, v)


def dddot(u: FieldOrArray, v: FieldOrArray):
    """Triple dot product."""
    return np.einsum('ijk...,ijk...', u, v)


def prod(u: FieldOrArray,
         v: FieldOrArray,
         w: Optional[FieldOrArray] = None):
    """Tensor product."""
    if w is None:
        return np.einsum('i...,j...->ij...', u, v)
    return np.einsum('i...,j...,k...->ijk...', u, v, w)


def mul(A: FieldOrArray, x: FieldOrArray):
    """Matrix multiplication."""
    return np.einsum('ij...,j...->i...', A, x)


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
    if N is None:
        if len(w.shape) > 2:
            N = w.shape[-3]
        else:
            raise ValueError("Cannot deduce the size of the identity matrix. "
                             "Give an explicit keyword argument N.")
    return eye(np.ones(w.shape[-2:]), N)


def det(A):
    """Determinant of an array `A` over trailing axis (if any)."""
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
    """Inverse of an array `A` over trailing axis (if any)."""
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
