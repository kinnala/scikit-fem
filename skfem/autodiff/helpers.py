import jax.numpy as jnp

from skfem import DiscreteField


def dot(u, v):
    return jnp.einsum('i...,i...', u, v)


def grad(u):
    if isinstance(u, DiscreteField):
        return u.grad
    return u[1]
