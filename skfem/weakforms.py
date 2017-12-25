# -*- coding: utf-8 -*-
"""
Weak forms for various PDE's
"""
def laplace(du, dv):
    if len(du)==1:
        return du*dv
    elif len(du)==2:
        return du[0]*dv[0] + du[1]*dv[1]
    elif len(du)==3:
        return du[0]*dv[0] + du[1]*dv[1] + du[2]*dv[2]
    else:
        raise NotImplementedError("Laplace weakform not implemented for the used dimension.")
