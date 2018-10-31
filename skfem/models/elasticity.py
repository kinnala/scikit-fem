"""Linear elasticity."""

import numpy as np
from skfem.assembly import bilinear_form, linear_form

def lame_parameters(E, nu):
    return E/(2.0*(1.0 + nu)), E*nu/((1.0 + nu)*(1.0 - 2.0*nu))

def linear_elasticity(Lambda=1.0, Mu=1.0):
    @bilinear_form
    def weakform(u, du, v, dv, w):
        from .helpers import ddot, trace, transpose, eye
        
        def C(T):
            return 2.0*Mu*T + Lambda*eye(trace(T), T.shape[0])

        def Eps(dw):
            return 0.5*(dw + transpose(dw))

        return ddot(C(Eps(du)), Eps(dv))

    return weakform
