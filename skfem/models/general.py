"""Bilinear and linear forms too general to put into a specific model."""

from skfem.assembly import bilinear_form


@bilinear_form
def divergence(u, du, v, dv, w):
    from .helpers import div    
    return div(du) * v
