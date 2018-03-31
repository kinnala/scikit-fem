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

def lame_parameters(E, nu):
    return E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

def linear_elasticity(Lambda=1.0, Mu=1.0):
    @bilinear_form
    def weakform(u, du, v, dv, w):
        def ddot(A, B):
            return A[0, 0] * B[0, 0] + \
                   A[0, 1] * B[0, 1] + \
                   A[1, 0] * B[1, 0] + \
                   A[1, 1] * B[1, 1] + \
                   A[0, 2] * B[0, 2] + \
                   A[2, 0] * B[2, 0] + \
                   A[1, 2] * B[1, 2] + \
                   A[2, 1] * B[2, 1] + \
                   A[2, 2] * B[2, 2]

        def tr(T):
            return T[0, 0] + T[1, 1] + T[2, 2]

        def C(T):
            return np.array([[2*Mu*T[0, 0] + Lambda*tr(T), 2*Mu*T[0, 1], 2*Mu*T[0, 2]],
                             [2*Mu*T[1, 0], 2*Mu*T[1, 1] + Lambda*tr(T), 2*Mu*T[1, 2]],
                             [2*Mu*T[2, 0], 2*Mu*T[2, 1], 2*Mu*T[2, 2] + Lambda*tr(T)]])

        def Eps(dw):
            return np.array([[dw[0][0], 0.5*(dw[0][1] + dw[1][0]), 0.5*(dw[0][2] + dw[2][0])],
                             [0.5*(dw[1][0] + dw[0][1]), dw[1][1], 0.5*(dw[1][2] + dw[2][1])],
                             [0.5*(dw[2][0] + dw[0][2]), 0.5*(dw[2][1] + dw[1][2]), dw[2][2]]])

        return ddot(C(Eps(du)), Eps(dv))

    return weakform
