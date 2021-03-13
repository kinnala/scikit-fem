"""Bilinear and linear forms too general to put into a specific model."""

from skfem.assembly import BilinearForm, LinearForm
from skfem.helpers import dot, div, curl


@BilinearForm
def divu(u, v, w):
    return div(u) * v


divergence = divu


@BilinearForm
def curluv(u, v, w):
    return dot(curl(u), v)


@LinearForm
def rot(v, w):
    return dot(curl(v), w['w'])


@LinearForm
def vrot(v, w):
    return dot(v, curl(w['w']))
