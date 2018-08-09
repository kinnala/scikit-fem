# -*- coding: utf-8 -*-
"""
Linear elasticity
"""
import numpy as np
from skfem.assembly import bilinear_form, linear_form

def lame_parameters(E, nu):
    return E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

def linear_elasticity(Lambda=1.0, Mu=1.0):
    @bilinear_form
    def weakform(u, du, v, dv, w):
        def ddot(A, B):
            return np.einsum('ij...,ij...', A, B)

        def tr(T):
            return np.einsum('ii...', T)

        def C(T):
            return 2*Mu*T + Lambda*np.einsum('ij,...->ij...', np.identity(T.shape[0]), tr(T))

        def Eps(dw):
            return 0.5*(dw + np.einsum('ij...->ji...', dw))

        return ddot(C(Eps(du)), Eps(dv))

    return weakform
