"""Bilinear and linear forms too general to put into a specific model."""

from skfem.assembly import BilinearForm, LinearForm
from .helpers import div, grad

import numpy as np


@BilinearForm
def divergence(u, v, w):
    return div(u) * v


@LinearForm
def rot(v, w):
    return np.einsum('i...,ij...,j...',
                     w.w, np.array([[0, 1], [-1, 0]]), grad(v))


@LinearForm
def vrot(v, w):
    return np.einsum('i...,ij...,j...',
                     v, np.array([[0, 1], [-1, 0]]), grad(w['w']))
