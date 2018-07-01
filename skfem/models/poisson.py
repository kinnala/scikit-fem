# -*- coding: utf-8 -*-
"""
Poisson equation
"""
import numpy as np
from skfem.assembly import bilinear_form, linear_form

@bilinear_form
def laplace(u, du, v, dv, w):
    if du.shape[0] == 2:
        return du[0]*dv[0] + du[1]*dv[1]
    elif du.shape[0] == 3:
        return du[0]*dv[0] + du[1]*dv[1] + du[2]*dv[2]
    else:
        raise NotImplementedError("Laplace weakform not implemented for the used dimension.")

@bilinear_form
def mass(u, du, v, dv, w):
    return u*v

@linear_form
def unit_load(v, dv, w):
    return v

def boundary_zero(ind):
    def bc(basis):
        return basis.find_dofs(ind)
    return bc
