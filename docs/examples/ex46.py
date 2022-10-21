"""
Like http://diyhpl.us/~nmz787/pdf/Using_the_FEniCS_Package_for_FEM_Solutions_in_Electromagnetics.pdf (4.1)
"""

import numpy as np

from skfem import *
from skfem.helpers import *


mesh = MeshTri().init_tensor(np.linspace(0, 1, 9),
                             np.linspace(0, .5, 5)).refined(3)
#basis = Basis(mesh, ElementTriN0() * ElementTriP1())
basis = Basis(mesh, ElementTriN0())

epsilon = lambda x: 1. + 0. * x[0]
# epsilon = lambda x: 3 * (x[1] < 0.25) + 1
one_over_u_r = 1


@BilinearForm
def aform(E, v, w):
    return one_over_u_r * curl(E) * curl(v)


@BilinearForm
def bform(E, v, w):
    return epsilon(w.x) * dot(E, v)


A = aform.assemble(basis)
B = bform.assemble(basis)

lams, xs = solve(*condense(A, B, D=basis.get_dofs()),
                 solver=solver_eigen_scipy_sym(k=6))


for itr, lam in enumerate(lams):
    if lam < 1e-8:
        continue
    basis.plot(xs[:, itr], colorbar=True).show()


print('TE10 (calculated): {}'.format(lams[-1]))
print('TE10 (analytical): {}'.format(np.pi ** 2))
