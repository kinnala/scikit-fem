# -*- coding: utf-8 -*-
"""
Weak forms for various PDE's
"""
import numpy as np

def laplace(du, dv):
    if len(du)==1:
        return du*dv
    elif len(du)==2:
        return du[0]*dv[0] + du[1]*dv[1]
    elif len(du)==3:
        return du[0]*dv[0] + du[1]*dv[1] + du[2]*dv[2]
    else:
        raise NotImplementedError("Laplace weakform not implemented for the used dimension.")

def elasticity_plane_strain(Lambda=1.0, Mu=1.0):
    def weakform(du, dv):
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