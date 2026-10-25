"""Statically indeterminate truss."""

import numpy as np
from skfem import *
from skfem.helpers import *


m = MeshLine(np.array([[0, 0],
                       [4, 0],
                       [8, 0],
                       [0, 4],
                       [4, 4]]).T,
             np.array([[0, 1],
                       [1, 2],
                       [0, 4],
                       [1, 3],
                       [2, 4],
                       [1, 4],
                       [3, 4]]).T)
basis = Basis(m, ElementVector(ElementLineP1(), dim=2))


@BilinearForm
def truss(u, v, w):
    return dot(dot(grad(u), w.t), w.t) * dot(dot(grad(v), w.t), w.t)


K = truss.assemble(basis)
F = basis.zeros()

# apply downwards force -60 kN to node nr 2
F[basis.get_dofs(nodes=[2]).all('u^2')] = -60

# fix nodes 0 and 3
D = basis.get_dofs(nodes=[0, 3]).all()

# solve displacements
u = solve(*condense(K, F, D=D))

# calculate axial forces

@Functional
def axial(w):
    return dot(dot(grad(w.u), w.t), w.t)


print(axial.elemental(basis, u=u) / basis.mesh_parameters()[:, 0])

