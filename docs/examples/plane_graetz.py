from skfem import *
from skfem.models.poisson import laplace

from math import ceil

import numpy as np

mesh_inlet_n = 2**5
height = 1.
length = 10.
peclet = 1e2

basis = InteriorBasis(
    (MeshLine(np.linspace(0, length,
                          ceil(mesh_inlet_n / height * length)))
     * MeshLine(np.linspace(0, height / 2, mesh_inlet_n)))._splitquads(),
    ElementTriP2())


@bilinear_form
def advection(u, du, v, dv, w):
    _, y = w.x
    velocity_0 = 6 * y * (height - y)  # parabolic plane Poiseuille
    return v * velocity_0 * du[0]


dofs = {'inlet': basis.get_dofs(lambda x, y: x == 0.),
        'floor': basis.get_dofs(lambda x, y: y == 0.)}
D = np.concatenate([d.all() for d in dofs.values()])
interior = basis.complement_dofs(D)

A = asm(laplace, basis) + peclet * asm(advection, basis)
t = np.zeros(basis.N)
t[dofs['floor'].all()] = 1.
t[interior] = solve(*condense(A, np.zeros_like(t), t, D=D))


if __name__ == '__main__':

    from pathlib import Path

    basis.mesh.plot(t[basis.nodal_dofs.flatten()], edgecolors='none')
    basis.mesh.savefig(Path(__file__).with_suffix('.png'),
                       bbox_inches='tight', pad_inches=0)
