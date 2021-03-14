"""Poisson equation."""

from skfem.assembly import BilinearForm, LinearForm
from skfem.helpers import grad, dot, ddot


@BilinearForm
def laplace(u, v, w):
    return dot(grad(u), grad(v))


@BilinearForm
def vector_laplace(u, v, w):
    return ddot(grad(u), grad(v))


@BilinearForm
def mass(u, v, w):
    return u * v


@LinearForm
def unit_load(v, w):
    return v
