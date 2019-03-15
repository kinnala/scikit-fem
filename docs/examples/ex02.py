from pathlib import Path
from skfem import *
import numpy as np

m = MeshTri.init_symmetric()
m.refine(3)

e = ElementTriMorley()
mapping = MappingAffine(m)
ib = InteriorBasis(m, e, mapping, 2)

@bilinear_form
def bilinf(u, du, ddu, v, dv, ddv, w):
    d = 0.1
    E = 200e9
    nu = 0.3

    def C(T):
        trT = T[0,0] + T[1,1]
        return np.array([[E/(1.0+nu)*(T[0, 0]+nu/(1.0-nu)*trT), E/(1.0+nu)*T[0, 1]],
                         [E/(1.0+nu)*T[1, 0], E/(1.0+nu)*(T[1, 1]+nu/(1.0-nu)*trT)]])

    def Eps(ddw):
        return np.array([[ddw[0][0], ddw[0][1]],
                         [ddw[1][0], ddw[1][1]]])

    def ddot(T1, T2):
        return T1[0, 0]*T2[0, 0] +\
               T1[0, 1]*T2[0, 1] +\
               T1[1, 0]*T2[1, 0] +\
               T1[1, 1]*T2[1, 1]

    return d**3/12.0*ddot(C(Eps(ddu)), Eps(ddv))

@linear_form
def linf(v, dv, ddv, w):
    return 1e6*v

K = asm(bilinf, ib)
f = asm(linf, ib)

boundary = {
        'left':  m.facets_satisfying(lambda x, y: x==0),
        'right': m.facets_satisfying(lambda x, y: x==1),
        'top':   m.facets_satisfying(lambda x, y: y==1),
        }

dofs = ib.get_dofs(boundary)

D = np.concatenate((
        dofs['left'].nodal['u'],
        dofs['left'].facet['u_n'],
        dofs['right'].nodal['u'],
        dofs['top'].nodal['u'],
        ))

I = ib.complement_dofs(D)
x = np.zeros_like(f)
x[I] = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    M, X = ib.refinterp(x, 3)
    ax = m.draw()
    M.plot(X, smooth=True, ax=ax, colorbar=True)
    M.savefig(Path(__file__).stem + '_solution.png')
