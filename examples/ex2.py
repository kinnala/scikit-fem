from skfem import *
import numpy as np

"""
Solve

  u'''' = 1
  u(0)=u'(0)=u(1)=u'(1)=0
  
using cubic Hermite elements.
"""

m = MeshLine()
m.refine(3)

e = ElementGlobalLineHermite()
a = AssemblerGlobal(m, e)

K = a.iasm(lambda ddu,ddv: ddu*ddv)
f = a.iasm(lambda v: 1*v)

def solve_cholmod(A, b):
    from sksparse.cholmod import cholesky
    factor = cholesky(A)
    return factor(b)

D = np.array([0, 1, 2, 3])

x = direct(K, f, D=D, solve=solve_cholmod)

M, X = a.refinterp(x, 3)

ax = m.plot(x[a.dofnum_u.n_dof[0, :]], color='ko')
M.plot(X, color='k-', ax=ax)
M.show()
