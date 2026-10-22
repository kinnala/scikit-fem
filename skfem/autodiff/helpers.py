"""Helper functions for defining nonlinear forms.

The helpers work both for the dual numbers passed to
:class:`~skfem.autodiff.NonlinearForm` and for plain arrays such as
:class:`~skfem.element.DiscreteField`.

"""
import numpy as np

from .dual import Dual, stack


def grad(u):
    """Gradient."""
    return u.grad


def div(u):
    """Divergence."""
    if u.div is not None:
        return u.div
    elif u.grad is not None:
        if len(u.grad.shape) == 4:
            return np.einsum('ii...', u.grad)
        return u.grad[0]  # one-dimensional u
    raise NotImplementedError


def curl(u):
    """Curl."""
    if u.curl is not None:
        return u.curl
    elif u.grad is not None:
        if len(u.grad.shape) == 3 and u.grad.shape[0] == 2:
            # curl of scalar field
            return stack([u.grad[1], -u.grad[0]])
        elif len(u.grad.shape) == 4 and u.grad.shape[0] == 2:
            # 2d-curl of vector field
            return u.grad[1, 0] - u.grad[0, 1]
        elif len(u.grad.shape) == 4 and u.grad.shape[0] == 3:
            return stack([
                u.grad[2, 1] - u.grad[1, 2],
                u.grad[0, 2] - u.grad[2, 0],
                u.grad[1, 0] - u.grad[0, 1],
            ])
    raise NotImplementedError


def d(u):
    """Gradient, divergence or curl."""
    if u.grad is not None:
        return u.grad
    elif u.div is not None:
        return u.div
    elif u.curl is not None:
        return u.curl
    raise NotImplementedError


def sym_grad(u):
    """Symmetric gradient."""
    return .5 * (u.grad + transpose(u.grad))


def dd(u):
    """Hessian."""
    return u.hess


def dot(u, v):
    """Dot product."""
    return np.einsum('i...,i...', u, v)


def ddot(u, v):
    """Double dot product."""
    return np.einsum('ij...,ij...', u, v)


def dddot(u, v):
    """Triple dot product."""
    return np.einsum('ijk...,ijk...', u, v)


def inner(u, v):
    """Inner product between any matching tensors."""
    if len(u.shape) == 2:
        return u * v
    elif len(u.shape) == 3:
        return dot(u, v)
    elif len(u.shape) == 4:
        return ddot(u, v)
    raise NotImplementedError


def prod(u, v, w=None):
    """Tensor product."""
    if w is None:
        return np.einsum('i...,j...->ij...', u, v)
    return np.einsum('i...,j...,k...->ijk...', u, v, w)


def mul(A, B):
    """Matrix-matrix or matrix-vector product."""
    if len(A.shape) == len(B.shape):
        return np.einsum('ij...,jk...->ik...', A, B)
    return np.einsum('ij...,j...->i...', A, B)


def trace(T):
    """Trace of matrix."""
    return np.einsum('ii...', T)


def transpose(T):
    """Transpose of matrix."""
    return np.einsum('ij...->ji...', T)


def eye(w, size):
    """Create diagonal matrix with w on diagonal."""
    zero = np.zeros_like(w.val if isinstance(w, Dual) else np.asarray(w))
    return stack([stack([w if i == j else zero for j in range(size)])
                  for i in range(size)])


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
    """Determinant of matrix."""
    if A.shape[0] == 3:
        return (A[0, 0] * (A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1])
                - A[0, 1] * (A[1, 0] * A[2, 2] - A[1, 2] * A[2, 0])
                + A[0, 2] * (A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0]))
    elif A.shape[0] == 2:
        return A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    elif A.shape[0] == 1:
        return A[0, 0]
    raise NotImplementedError


def inv(A):
    """Inverse of matrix."""
    detA = det(A)
    if A.shape[0] == 3:
        return stack([
            stack([A[1, 1] * A[2, 2] - A[1, 2] * A[2, 1],
                   A[0, 2] * A[2, 1] - A[0, 1] * A[2, 2],
                   A[0, 1] * A[1, 2] - A[0, 2] * A[1, 1]]),
            stack([A[1, 2] * A[2, 0] - A[1, 0] * A[2, 2],
                   A[0, 0] * A[2, 2] - A[0, 2] * A[2, 0],
                   A[0, 2] * A[1, 0] - A[0, 0] * A[1, 2]]),
            stack([A[1, 0] * A[2, 1] - A[1, 1] * A[2, 0],
                   A[0, 1] * A[2, 0] - A[0, 0] * A[2, 1],
                   A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]]),
        ]) / detA
    elif A.shape[0] == 2:
        return stack([stack([A[1, 1], -A[0, 1]]),
                      stack([-A[1, 0], A[0, 0]])]) / detA
    elif A.shape[0] == 1:
        return stack([stack([1. / A[0, 0]])])
    raise NotImplementedError


def cross(A, B):
    """Cross product."""
    if A.shape[0] == 2:
        return A[0] * B[1] - A[1] * B[0]
    if A.shape[0] == 3:
        return stack([
            A[1] * B[2] - A[2] * B[1],
            A[2] * B[0] - A[0] * B[2],
            A[0] * B[1] - A[1] * B[0],
        ])
    raise NotImplementedError
