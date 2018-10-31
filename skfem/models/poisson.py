"""Poisson equation."""

import numpy as np
from skfem.assembly import bilinear_form, linear_form


@bilinear_form
def laplace(u, du, v, dv, w):
    return sum(du * dv)

@bilinear_form
def vector_laplace(u, du, v, dv, w):
    from .helpers import ddot
    return ddot(du, dv)

@bilinear_form
def mass(u, du, v, dv, w):
    return u * v

@linear_form
def unit_load(v, dv, w):
    return v
