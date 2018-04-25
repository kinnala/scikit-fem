"""
Solve the linear elastic eigenvalue problem.
"""

from skfem import *
from skfem.models.elasticity import *
from scipy.sparse.linalg import eigsh
import numpy as np

m1 = MeshLine(np.linspace(0,5,50))
m2 = MeshLine(np.linspace(0,1,10))
m = m1*m2

e1 = ElementQ1()

map = MappingIsoparametric(m, e1)

e = ElementVectorH1(e1)

gb = InteriorBasis(m, e, map, 2)

K = asm(plane_strain(1.0, 1.0), gb)

@bilinear_form
def mass(u, du, v, dv, w):
    return u[0]*v[0] + u[1]*v[1]

M = asm(mass, gb)

y, D = gb.find_dofs(lambda x, y: x==0.0)

I = gb.dofnum.complement_dofs(D)

L, x = eigsh(K[I].T[I].T, k=6, M=M[I].T[I].T, which='SM')

y[I] = x[:, 4]

MeshQuad(np.array([m.p[0, :]+y[gb.dofnum.n_dof[0, :]], m.p[1, :]+y[gb.dofnum.n_dof[1, :]]]), m.t).draw()
m.show()
