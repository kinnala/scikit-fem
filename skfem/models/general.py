"""Bilinear and linear forms too general to put into a specific model."""

from skfem.assembly import BilinearForm, linear_form
from .helpers import div


@BilinearForm
def divergence(u, v, w):
    return div(u) * v


@linear_form
def rot(v, dv, w):
    return dv[1] * w.w[0] - dv[0] * w.w[1]
