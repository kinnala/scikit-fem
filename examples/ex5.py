from skfem import *

"""
Impose integral condition using a Lagrange multiplier
"""

m = MeshTri()
for itr in range(5):
    m.refine()

a = AssemblerLocal(m, ElementLocalTriP1())
A = a.iasm(lambda du, dv: du[0]*dv[0] + du[1]*dv[1])
B = a.fasm(lambda du, v, n, x: -(du[0]*n[0] + du[1]*n[1])*v*(x[0]==1.0))

b = a.fasm(lambda dv, n, x: -(dv[0]*n[0] + dv[1]*n[1])*(x[0]==1.0))

import scipy.sparse
b = scipy.sparse.csr_matrix(b)
K = scipy.sparse.bmat([[A+B, b.T], [b, None]]).tocsr()

_, d = a.essential_bc(lambda x, y: (y==0.0))

import numpy as np
f = np.concatenate((np.zeros(A.shape[0]), -1.0*np.ones(1)))

x = direct(K, f, D=d)

m.plot3(x[:-1])
m.show()
