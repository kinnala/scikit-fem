from skfem import *
from skfem.weakforms import elasticity_plane_strain
from scipy.sparse.linalg import eigsh
import numpy as np

"""
Solve linear elastic eigenvalue problem.
"""

m1 = MeshLine(np.linspace(0,5,50))
m2 = MeshLine(np.linspace(0,1,10))
m = m1*m2

e1 = ElementLocalQ1()
e = ElementLocalH1Vec(e1)
a = AssemblerLocal(m, e)

K = a.iasm(elasticity_plane_strain(1.0, 1.0))
M = a.iasm(lambda u, v: u[0]*v[0] + u[1]*v[1])

y, D = a.essential_bc(lambda x, y: x==0.0)

I = a.dofnum_u.complement_dofs(D)

L, x = eigsh(K[I].T[I].T, k=6, M=M[I].T[I].T, which='SM')

y[I] = x[:, 4]

MeshQuad(np.array([m.p[0, :]+y[a.dofnum_u.n_dof[0, :]], m.p[1, :]+y[a.dofnum_u.n_dof[1, :]]]), m.t).draw()
m.show()