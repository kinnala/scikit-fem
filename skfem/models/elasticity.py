# -*- coding: utf-8 -*-
"""
Linear elasticity
"""
import numpy as np
from skfem.assembly import bilinear_form, linear_form

def basis_elasticity(m):
    import skfem.mesh
    from skfem.assembly import InteriorBasis
    from skfem.element import ElementVectorH1, ElementHex1,\
            ElementTetP1, ElementTriP1
    from skfem.mapping import MappingAffine, MappingIsoparametric
    if type(m) is skfem.mesh.MeshHex:
        e1 = ElementHex1()
        e = ElementVectorH1(e1)
        map = MappingIsoparametric(m, e1)
        return InteriorBasis(m, e, map, 3)
    elif type(m) is skfem.mesh.MeshTet:
        e1 = ElementTetP1()
        e = ElementVectorH1(e1)
        map = MappingAffine(m)
        return InteriorBasis(m, e, map, 2)
    elif type(m) is skfem.mesh.MeshTri:
        e1 = ElementTriP1()
        e = ElementVectorH1(e1)
        map = MappingAffine(m)
        return InteriorBasis(m, e, map, 2)
    else:
        raise NotImplementedError("basis_elasticity does not support the given mesh.")

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

def boundary_prescribed(x=None, y=None, z=None):
    """Supports only 1 DOF/node elements."""
    def prescribed(ind):
        nonlocal x, y, z
        def bc(basis):
            nonlocal x, y, z
            w = np.zeros(basis.dofnum.N)
            D = np.array([])
            if x is not None:
                if type(x) is int or type(x) is float:
                    X = lambda a, b, c: 0*a + x
                else:
                    X = x
                x1, D1 = basis.find_dofs(ind, bc=X, dofrows=[0], check_facets=False, check_edges=False)
                w += x1
                D = np.concatenate((D, D1))
            if y is not None:
                if type(y) == int or float:
                    Y = lambda a, b, c: 0*a + y
                else:
                    Y = y
                x2, D2 = basis.find_dofs(ind, bc=Y, dofrows=[1], check_facets=False, check_edges=False)
                w += x2
                D = np.concatenate((D, D2))
            if z is not None:
                if type(z) == int or float:
                    Z = lambda a, b, c: 0*a + z
                else:
                    Z = z
                x3, D3 = basis.find_dofs(ind, bc=Z, dofrows=[2], check_facets=False, check_edges=False)
                w += x3
                D = np.concatenate((D, D3))
            return w, D
        return bc
    return prescribed

def boundary_clamped(ind):
    def bc(basis):
        return basis.find_dofs(ind)
    return bc

def linear_elasticity(Lambda=1.0, Mu=1.0):
    @bilinear_form
    def weakform(u, du, v, dv, w):
        def ddot(A, B):
            return A[0, 0] * B[0, 0] + \
                   A[0, 1] * B[0, 1] + \
                   A[1, 0] * B[1, 0] + \
                   A[1, 1] * B[1, 1] + \
                   A[0, 2] * B[0, 2] + \
                   A[2, 0] * B[2, 0] + \
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
                             [0.5*(dw[1][0] + dw[0][1]), dw[1][1], 0.5*(dw[1][2] + dw[2][1])],
                             [0.5*(dw[2][0] + dw[0][2]), 0.5*(dw[2][1] + dw[1][2]), dw[2][2]]])

        return ddot(C(Eps(du)), Eps(dv))

    return weakform
