# -*- coding: utf-8 -*-
"""
Weak forms for various PDE's
"""
import numpy as np
from skfem.utils import bilinear_form, linear_form

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

def plane_strain(Lambda=1.0, Mu=1.0):
    @bilinear_form
    def weakform(u, du, v, dv, w):
        def ddot(A, B):
            return A[0, 0] * B[0, 0] + \
                   A[0, 1] * B[0, 1] + \
                   A[1, 0] * B[1, 0] + \
                   A[1, 1] * B[1, 1]

        def tr(T):
            return T[0, 0] + T[1, 1]

        def C(T):
            return np.array([[2*Mu*T[0, 0] + Lambda*tr(T), 2*Mu*T[0, 1]],
                             [2*Mu*T[1, 0], 2*Mu*T[1, 1] + Lambda*tr(T)]])

        def Eps(dw):
            return np.array([[dw[0][0], 0.5*(dw[0][1] + dw[1][0])],
                             [0.5*(dw[1][0] + dw[0][1]), dw[1][1]]])

        return ddot(C(Eps(du)), Eps(dv))

    return weakform
