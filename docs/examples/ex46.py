"""
Like http://diyhpl.us/~nmz787/pdf/Using_the_FEniCS_Package_for_FEM_Solutions_in_Electromagnetics.pdf (4.1)
"""

import numpy as np
import matplotlib.pyplot as plt

import skfem
from skfem import Basis, ElementTriP1, ElementTriN0, asm, BilinearForm, solve, solver_eigen_scipy, condense, \
    ElementComposite
from skfem.helpers import curl, grad, inner
from skfem.visuals.matplotlib import draw, plot

mesh = skfem.MeshTri().init_tensor(np.linspace(0, 1, 30), np.linspace(0, .5, 30))
basis = Basis(mesh, ElementComposite(ElementTriN0(), ElementTriP1()))

epsilon = 1
one_over_u_r = 1


@BilinearForm
def s_tt_ij(N_i, L_i, N_j, L_j, w):
    return one_over_u_r * inner(curl(N_i), curl(N_j))


@BilinearForm
def t_tt_ij(N_i, L_i, N_j, L_j, w):
    return epsilon * inner(N_i, N_j)


@BilinearForm
def s_zz_ij(N_i, L_i, N_j, L_j, w): \
        return one_over_u_r * inner(grad(L_i), grad(L_j))


@BilinearForm
def t_zz_ij(N_i, L_i, N_j, L_j, w):
    return epsilon * L_i * L_j


s_ij = asm(s_tt_ij, basis) + asm(s_zz_ij, basis)
t_ij = asm(t_tt_ij, basis) + asm(t_zz_ij, basis)

eigenValues, eigenVectors = solve(*condense(s_ij, t_ij, D=mesh.boundary_nodes()),
                                  solver=solver_eigen_scipy(k=6, sigma=1, which='LM'))

print(eigenValues)

plot(basis.split_bases()[0], eigenVectors[basis.split_indices()[0], 0], ax=draw(mesh, boundaries_only=True),
     colorbar=True)
plt.show()

plot(basis.split_bases()[1], eigenVectors[basis.split_indices()[1], 0], ax=draw(mesh, boundaries_only=True),
     colorbar=True)
plt.show()

print(np.max(eigenVectors[basis.split_indices()[0]]), np.max(eigenVectors[basis.split_indices()[1]]))
