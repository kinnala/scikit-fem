import jax.numpy as jnp

from skfem import DiscreteField


def dot(u, v):
    return jnp.einsum('i...,i...', u, v)


def ddot(u, v):
    return jnp.einsum('ij...,ij...', u, v)


def grad(u):
    if isinstance(u, DiscreteField):
        return u.grad
    return u[1]


def dd(u):
    if isinstance(u, DiscreteField):
        return u.hess
    return u[4]


def transpose(T):
    return jnp.einsum('ij...->ji...', T)


def mul(A, B):
    if len(A.shape) == len(B.shape):
        return jnp.einsum('ij...,jk...->ik...', A, B)
    return jnp.einsum('ij...,j...->i...', A, B)


def trace(T):
    return jnp.einsum('ii...', T)


def eye(w, size):
    return jnp.array([[w if i == j else 0. * w for i in range(size)]
                      for j in range(size)])
