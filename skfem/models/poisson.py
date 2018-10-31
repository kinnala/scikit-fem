# -*- coding: utf-8 -*-
"""
Poisson equation
"""
from skfem.assembly import bilinear_form, linear_form

import numpy as np

@bilinear_form
def laplace(u, du, v, dv, w):
    return sum(du*dv)

@bilinear_form
def vector_laplace(u, du, v, dv, w):
    return np.einsum('ij...,ij...', du, dv)

@bilinear_form
def mass(u, du, v, dv, w):
    return u*v

@linear_form
def unit_load(v, dv, w):
    return v
