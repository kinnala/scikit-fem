"""Poisson equation."""

from cudaskfem.assembly import BilinearForm, LinearForm
from cudaskfem.helpers import grad, dot, ddot


@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


@BilinearForm
def vector_laplace(u, v, _):
    return ddot(grad(u), grad(v))


@BilinearForm
def mass(u, v, _):
    return u * v


@LinearForm
def unit_load(v, _):
    return v
