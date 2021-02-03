"""Bilinear and linear forms too general to put into a specific model."""

from skfem.assembly import BilinearForm, LinearForm
from .helpers import dot, div, curl


@BilinearForm
def divergence(u, v, w):
    return div(u) * v


@LinearForm
def rot(v, w):
    return dot(curl(v), w["w"])


@LinearForm
def vrot(v, w):
    return dot(v, curl(w["w"]))
