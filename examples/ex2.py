from skfem import *
import numpy as np

m = MeshTri()
m.refine(3)

e = ElementMorley()
map = MappingAffine(m)
ib = InteriorBasis(m, e, map, 4)

@bilinear_form
def bilinf(u, du, v, dv, w):
    # plate thickness
    d = 1.0
    E = 1.0
    nu = 0.3

    def C(T):
        trT=T[0,0]+T[1,1]
        return np.array([[E/(1.0+nu)*(T[0, 0]+nu/(1.0-nu)*trT), E/(1.0+nu)*T[0, 1]],
                         [E/(1.0+nu)*T[1, 0], E/(1.0+nu)*(T[1, 1]+nu/(1.0-nu)*trT)]])

    def Eps(ddU):
        return np.array([[ddU[0][0], ddU[0][1]],
                         [ddU[1][0], ddU[1][1]]])

    def ddot(T1,T2):
        return T1[0, 0]*T2[0, 0]+\
               T1[0, 1]*T2[0, 1]+\
               T1[1, 0]*T2[1, 0]+\
               T1[1, 1]*T2[1, 1]

    ddu = du[1]
    ddv = dv[1]

    return d**3/12.0*ddot(C(Eps(ddu)), Eps(ddv))

K = asm(bilinf, ib)
f = asm(unit_load, ib)

x, D = ib.essential_bc()
I = ib.dofnum.complement_dofs(D)

x[I] = solve(*condense(K, f, I=I), solver=solver_direct_cholmod())

M, X = ib.refinterp(x, 3)
ax = m.draw()
M.plot(X, smooth=True, edgecolors='', ax=ax)
M.show()
