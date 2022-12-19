import autograd.numpy as np
from autograd.builtins import isinstance

from skfem import DiscreteField


def dot(u, v):
    if isinstance(u, tuple):
        u = u[0]
    if isinstance(v, tuple):
        v = v[0]
    return np.einsum('i...,i...', u, v)


def ddot(u, v):
    if isinstance(u, tuple):
        u = u[0]
    if isinstance(v, tuple):
        v = v[0]
    return np.einsum('ij...,ij...', u, v)


def dddot(u, v):
    if isinstance(u, tuple):
        u = u[0]
    if isinstance(v, tuple):
        v = v[0]
    return np.einsum('ijk...,ijk...', u, v)


def grad(u):
    if isinstance(u, DiscreteField):
        return u.grad
    return u[1]


def sym_grad(u):
    if isinstance(u, DiscreteField):
        return .5 * (u.grad + transpose(u.grad))
    return .5 * (u[1] + transpose(u[1]))


def div(u):
    if len(u[1].shape) == 4:
        return np.einsum('ii...', u[1])
    return u[2]


def dd(u):
    if isinstance(u, DiscreteField):
        return u.hess
    return u[4]


def transpose(T):
    if isinstance(T, tuple):
        T = T[0]
    return np.einsum('ij...->ji...', T)


def mul(A, B):
    if isinstance(A, tuple):
        A = A[0]
    if isinstance(B, tuple):
        B = B[0]
    if len(A.shape) == len(B.shape):
        return np.einsum('ij...,jk...->ik...', A, B)
    return np.einsum('ij...,j...->i...', A, B)


def trace(T):
    if isinstance(T, tuple):
        T = T[0]
    return np.einsum('ii...', T)


def eye(w, size):
    return np.array([[w if i == j else 0. * w for i in range(size)]
                     for j in range(size)])


def det(A):
    detA = np.zeros_like(A[0, 0])
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
