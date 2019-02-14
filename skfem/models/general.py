"""Bilinear and linear forms too general to put into a specific model."""

from skfem.assembly import bilinear_form, linear_form


@bilinear_form
def divergence(u, du, v, dv, w):
    from .helpers import div    
    return div(du) * v


@linear_form
def rot(v, dv, w):
    return dv[1] * w.w[0] - dv[0] * w.w[1]
