"""
Like http://diyhpl.us/~nmz787/pdf/Using_the_FEniCS_Package_for_FEM_Solutions_in_Electromagnetics.pdf (4.1)
"""

import numpy as np

from skfem import *
from skfem.helpers import *


mesh = MeshTri().init_tensor(np.linspace(0, 1, 9),
                             np.linspace(0, .5, 5)).refined()
basis = Basis(mesh, ElementTriN0() * ElementTriP1())

epsilon = 1
one_over_u_r = 1


def rot(x):
    return np.array([-x[1], x[0]])


@BilinearForm
def aform(Et, Ez, vt, vz, w):
    return one_over_u_r * (curl(Et) * curl(vt) + dot(grad(Ez), grad(vz)))


@BilinearForm
def bform(Et, Ez, vt, vz, w):
    return epsilon * (dot(Et, vt) + Ez * vz)


A = aform.assemble(basis)
B = bform.assemble(basis)

lam, x = solve(*condense(A, B, D=basis.get_dofs()),
               solver=solver_eigen_scipy_sym(k=6))


for itr in range(6):
    (Et, Etbasis), (Ez, Ezbasis) = basis.split(x[:, itr])
    Etbasis.plot(Et).show()
    Ezbasis.plot(Ez, colorbar=True).show()

# basis.plot(basis.split_bases()[0],
#            eigenVectors[basis.split_indices()[0], 0],
#            ax=mesh.draw(boundaries_only=True),
#            colorbar=True).show()

# basis.plot(basis.split_bases()[1],
#            eigenVectors[basis.split_indices()[1], 0],
#            ax=mesh.draw(boundaries_only=True),
#            colorbar=True).show()
