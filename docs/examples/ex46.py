"""
Like http://diyhpl.us/~nmz787/pdf/Using_the_FEniCS_Package_for_FEM_Solutions_in_Electromagnetics.pdf (4.1)
"""

import numpy as np

from skfem import *
from skfem.helpers import *


mesh = MeshTri().init_tensor(np.linspace(0, 1, 10),
                             np.linspace(0, .5, 10))
basis = Basis(mesh, ElementTriN0() * ElementTriP1())

epsilon = 1
one_over_u_r = 1


@BilinearForm
def aform(Et, Ez, vt, vz, w):
    return one_over_u_r * (dot(curl(Et), curl(vt)) + dot(grad(Ez), grad(vz)))


@BilinearForm
def bform(Et, Ez, vt, vz, w):
    return epsilon * (dot(Et, vt) + Ez * vz)


A = aform.assemble(basis)
B = bform.assemble(basis)

lam, x = solve(*condense(A, B, D=basis.get_dofs()),
               solver=solver_eigen_scipy(k=6))

(Et, Etbasis), (Ez, Ezbasis) = basis.split(x[:, 2])

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
