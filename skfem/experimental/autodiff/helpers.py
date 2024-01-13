import jax.numpy as jnp

from skfem import DiscreteField


def dot(u, v):
    # if isinstance(u, tuple):
    #     u = u[0]
    # if isinstance(v, tuple):
    #     v = v[0]
    return jnp.einsum('i...,i...', u, v)


def ddot(u, v):
    # if isinstance(u, tuple):
    #     u = u[0]
    # if isinstance(v, tuple):
    #     v = v[0]
    return jnp.einsum('ij...,ij...', u, v)


def dddot(u, v):
    # if isinstance(u, tuple):
    #     u = u[0]
    # if isinstance(v, tuple):
    #     v = v[0]
    return jnp.einsum('ijk...,ijk...', u, v)


def grad(u):
    # if isinstance(u, DiscreteField):
    #     return u.grad
    return u.grad


def sym_grad(u):
    # if isinstance(u, DiscreteField):
    #     return .5 * (u.grad + transpose(u.grad))
    return .5 * (u.grad + transpose(u.grad))


def div(u):
    if len(u.grad.shape) == 4:
        return jnp.einsum('ii...', u.grad)
    return u.div


def dd(u):
    return u.hess


def transpose(T):
    return jnp.einsum('ij...->ji...', T)


def mul(A, B):
    if len(A.shape) == len(B.shape):
        return jnp.einsum('ij...,jk...->ik...', A, B)
    return jnp.einsum('ij...,j...->i...', A, B)


def trace(T):
    return jnp.einsum('ii...', T)


def eye(w, size):
    return jnp.array([[w if i == j else 0. * w
                       for i in range(size)]
                     for j in range(size)])


def det(A):
    detA = jnp.zeros_like(A[0, 0])
    if A.shape[0] == 3:
        detA = (A[0, 0] * (A[1, 1] * A[2, 2]
                           - A[1, 2] * A[2, 1])
                - A[0, 1] * (A[1, 0] * A[2, 2] -
                             - A[1, 2] * A[2, 0])
                + A[0, 2] * (A[1, 0] * A[2, 1]
                             - A[1, 1] * A[2, 0]))
    elif A.shape[0] == 2:
        detA = A[0, 0] * A[1, 1] - A[1, 0] * A[0, 1]
    return detA
