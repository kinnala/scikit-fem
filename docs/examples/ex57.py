"""Truss elements."""

import numpy as np
from skfem import *
from skfem.helpers import *


m = MeshLine(np.array([[0., 0.],
                       [.5, 0.],
                       [1., 0.],
                       [0., .5],
                       [.5, .5]]).T,
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

# apply downwards force -6 kN to node nr 2
F[basis.get_dofs(nodes=[2]).all('u^2')] = -6000

# fix nodes 0 and 3
D = basis.get_dofs(nodes=[0, 3]).all()

# solve displacements
u = solve(*condense(K, F, D=D))

# calculate axial forces

@Functional
def axial(w):
    return dot(dot(grad(w.u), w.t), w.t)


forces = np.round(axial.elemental(basis, u=u)
                  / basis.mesh_parameters()[:, 0]
                  / 1000, decimals=2)
ref = np.array([-8.65, -6.00, -4.73, 3.75, 8.49, -2.65, 9.35])

print("Calculated forces [kN]: {}".format(forces))
print("Reference forces [kN]: {}".format(ref))

