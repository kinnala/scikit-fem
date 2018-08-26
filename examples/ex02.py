"""
Author: kinnala

Solve the Kirchhoff plate bending problem in a unit square
with clamped boundary conditions using the nonconforming
Morley element. Demonstrates also the visualization of
higher order solutions using 'GlobalBasis.refinterp'.
"""

from skfem import *
import numpy as np

m = MeshTri()
m.refine(3)

e = ElementTriMorley()
map = MappingAffine(m)
ib = InteriorBasis(m, e, map, 4)

@bilinear_form
def bilinf(u, du, ddu, v, dv, ddv, w):
    # plate thickness
    d = 1.0
    E = 1.0
    nu = 0.3

    def C(T):
        trT = T[0,0] + T[1,1]
        return np.array([[E/(1.0+nu)*(T[0, 0]+nu/(1.0-nu)*trT), E/(1.0+nu)*T[0, 1]],
                         [E/(1.0+nu)*T[1, 0], E/(1.0+nu)*(T[1, 1]+nu/(1.0-nu)*trT)]])

    def Eps(ddU):
        return np.array([[ddU[0][0], ddU[0][1]],
                         [ddU[1][0], ddU[1][1]]])

    def ddot(T1, T2):
        return T1[0, 0]*T2[0, 0] +\
               T1[0, 1]*T2[0, 1] +\
               T1[1, 0]*T2[1, 0] +\
               T1[1, 1]*T2[1, 1]

    return d**3/12.0*ddot(C(Eps(ddu)), Eps(ddv))

@linear_form
def linf(v, dv, ddv, w):
    return 1.0*v

K = asm(bilinf, ib)
f = asm(linf, ib)

x, D = ib.find_dofs()
I = ib.complement_dofs(D)

x[I] = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    M, X = ib.refinterp(x, 3)
    ax = m.draw()
    M.plot(X, smooth=True, edgecolors='', ax=ax)
    M.show()
