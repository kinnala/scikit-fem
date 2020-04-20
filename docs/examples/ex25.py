from skfem import *
from skfem.models.poisson import laplace, mass

from math import ceil

import numpy as np

mesh_inlet_n = 2**5
height = 1.
length = 10.
peclet = 1e2

mesh = MeshQuad.init_tensor(
    np.linspace(0, length, ceil(mesh_inlet_n / height * length)),
    np.linspace(0, height / 2, mesh_inlet_n))
basis = InteriorBasis(mesh, ElementQuad2())


@BilinearForm
def advection(u, v, w):
    from skfem.helpers import grad
    _, y = w.x
    velocity_0 = 6 * y * (height - y)  # parabolic plane Poiseuille
    return v * velocity_0 * grad(u)[0]


dofs = basis.get_dofs({'inlet': lambda x: x[0] == 0.,
                       'floor': lambda x: x[1] == 0.})
interior = basis.complement_dofs(dofs)

A = asm(laplace, basis) + peclet * asm(advection, basis)
t = np.zeros(basis.N)
t[dofs['floor'].all()] = 1.
t = solve(*condense(A, np.zeros_like(t), t, I=interior))

basis0 = InteriorBasis(mesh, ElementQuad0(), quadrature=basis.quadrature)
t0 = solve(asm(mass, basis0),
           asm(mass, basis, basis0) @ t)


if __name__ == '__main__':
    from pathlib import Path
    from skfem.visuals.matplotlib import plot, savefig

    plot(mesh, t0)
    savefig(Path(__file__).with_suffix('.png'),
            bbox_inches='tight', pad_inches=0)
