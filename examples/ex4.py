from skfem import *
from skfem.weakforms import elasticity_plane_strain
from skfem.mesh import *
from skfem.assembly import *
from scipy.sparse.linalg import eigsh
from skfem.extern_sfepy import *
import numpy as np
import matplotlib.pyplot as plt

"""
Solve linear elastic contact problem using penalty method
"""

m = read_comsol("examples/square_smalltris.mphtxt")
M = read_comsol("examples/square_largetris.mphtxt")
M.translate((1.0, 0.0))

def rule(x, y):
    return (x==1.0)*(y+1.0)

mortar = MeshLineMortar(m, M, rule)

e1 = ElementLocalTriP1()
e = ElementLocalH1Vec(e1)

a1 = AssemblerLocal(m, e)
a2 = AssemblerLocal(M, e)
a12 = AssemblerLocalMortar(mortar, m, M, e)

E1 = 1000.0
E2 = 1000.0

nu1 = 0.3
nu2 = 0.3

Mu1 = E1/(2.0*(1.0 + nu1))
Mu2 = E2/(2.0*(1.0 + nu2))

Lambda1 = E1*nu1/((1.0 + nu1)*(1.0 - 2.0*nu1))
Lambda2 = E2*nu2/((1.0 + nu2)*(1.0 - 2.0*nu2))

weakform1 = elasticity_plane_strain(Lambda=Lambda1, Mu=Mu1)
weakform2 = elasticity_plane_strain(Lambda=Lambda2, Mu=Mu2)

K1 = a1.iasm(weakform1)
K2 = a2.iasm(weakform2)
K12 = a12.fasm(lambda u1, u2, v1, v2, n: ((u1[0]-u2[0])*n[0] + (u1[1]-u2[1])*n[1])*((v1[0]-v2[0])*n[0] + (v1[1]-v2[1])*n[1]))

f1 = a1.iasm(lambda v: -50*v[1])
f2 = a2.iasm(lambda v: 0*v[0])

import scipy.sparse
K = scipy.sparse.bmat([[K1, None],[None, K2]]).tocsr() + 20000*K12

i1 = np.arange(K1.shape[0])
i2 = np.arange(K2.shape[0]) + K1.shape[0]

_, D1 = a1.essential_bc(lambda x, y: x==0.0)
_, D2 = a2.essential_bc(lambda x, y: x==2.0)

x = np.zeros(K.shape[0])

f = np.hstack((f1, f2))

x = direct(K, f, D=np.concatenate((D1, D2 + a1.dofnum_u.N)))

sf = 1

m.p[0, :] = m.p[0, :] + sf*x[i1][a1.dofnum_u.n_dof[0, :]]
m.p[1, :] = m.p[1, :] + sf*x[i1][a1.dofnum_u.n_dof[1, :]]

M.p[0, :] = M.p[0, :] + sf*x[i2][a2.dofnum_u.n_dof[0, :]]
M.p[1, :] = M.p[1, :] + sf*x[i2][a2.dofnum_u.n_dof[1, :]]

ax = m.draw()
M.draw(ax=ax)
m.show()
