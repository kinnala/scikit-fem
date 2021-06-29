r"""Forced convection.

We begin the study of forced convection with the plane Graetz problem; viz. the steady distribution of temperature in a plane channel with zero inlet temperature and unit temperature on the walls and a steady laminar unidirectional parabolic plane-Poiseuille flow.

The governing advection–diffusion equation is

.. math::
   \mathrm{Pe} \;u\frac{\partial T}{\partial x} = \nabla^2 T
where the velocity profile is

.. math::
   u (y) = 6 y (1 - y), \qquad (0 < y < 1)
The equations here have been nondimensionalized by the width of the channel and the volumetric flow-rate.  The governing parameter is the Péclet number, being the mean velocity times the width divided by the thermal diffusivity.

Because the problem is symmetric about :math:`y = \frac{1}{2}`, only half is solved here, with natural boundary conditions along the centreline.

"""
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
basis = Basis(mesh, ElementQuad2())


@BilinearForm
def advection(u, v, w):
    from skfem.helpers import grad
    _, y = w.x
    velocity_0 = 6 * y * (height - y)  # parabolic plane Poiseuille
    return v * velocity_0 * grad(u)[0]


dofs = basis.find_dofs({'inlet': mesh.facets_satisfying(lambda x: x[0] == 0.),
                        'floor': mesh.facets_satisfying(lambda x: x[1] == 0.)})
interior = basis.complement_dofs(dofs)

A = asm(laplace, basis) + peclet * asm(advection, basis)
t = basis.zeros()
t[dofs['floor'].all()] = 1.
t = solve(*condense(A, x=t, I=interior))

basis0 = basis.with_element(ElementQuad0())
t0 = solve(asm(mass, basis0),
           asm(mass, basis, basis0) @ t)


if __name__ == '__main__':
    from pathlib import Path
    from skfem.visuals.matplotlib import plot, savefig

    plot(mesh, t0)
    savefig(Path(__file__).with_suffix('.png'),
            bbox_inches='tight', pad_inches=0)
