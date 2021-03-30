r"""Laplace with mixed boundary conditions.

This example is another extension of `ex01.py`, still solving the Laplace
equation but now with mixed boundary conditions, two parts isopotential (charged
and earthed) and the rest insulated.

The example is :math:`\Delta u = 0` in
:math:`\Omega=\{(x,y):1<x^2+y^2<4,~0<\theta<\pi/2\}`, where :math:`\tan \theta =
y/x`, with :math:`u = 0` on :math:`y = 0` and :math:`u = 1` on :math:`x =
0`.  The mesh is first constructed as a rectangle in the :math:`r\theta`-plane,
where the isopotential parts are conveniently tagged using `skfem.Mesh.with_boundaries`;
then the mesh is mapped to the :math:`xy`-plane.

The exact solution is :math:`u = 2 \theta / \pi`. The field strength is :math:`|\nabla u|^2 = 4 / \pi^2 (x^2 + y^2)`
so the conductance (for unit potential difference and conductivity) is
:math:`\|\nabla u\|^2 = 2 \ln 2 / \pi`.

"""

from skfem import *
from skfem.models.poisson import laplace, mass
from skfem.io import from_meshio

import numpy as np

radii = [1., 2.]
lcar = .1

mesh = (MeshTri
        .init_tensor(np.linspace(*radii, 1 + int(np.diff(radii) / lcar)),
                     np.linspace(0, np.pi/2, 1 + int(3*np.pi/4 / lcar)))
        .with_boundaries({
            'ground': lambda xi: xi[1] == 0.,
            'positive': lambda xi: xi[1] == np.pi/2,
        }))
mesh = mesh.translated(mesh.p[0] * np.stack([np.cos(mesh.p[1]),
                                             np.sin(mesh.p[1])]) - mesh.p)

elements = ElementTriP2()
basis = InteriorBasis(mesh, elements)
A = asm(laplace, basis)

boundary_dofs = basis.find_dofs()

u = np.zeros(basis.N)
u[boundary_dofs['positive'].all()] = 1.
u = solve(*condense(A, x=u, D=boundary_dofs))

M = asm(mass, basis)
u_exact = 2 * np.arctan2(*basis.doflocs[::-1]) / np.pi
u_error = u - u_exact
error_L2 = np.sqrt(u_error @ M @ u_error)
conductance = {'skfem': u @ A @ u,
               'exact': 2 * np.log(2) / np.pi}


@Functional
def port_flux(w):
    from skfem.helpers import dot, grad
    return dot(w.n, grad(w['u']))


current = {}
for port, boundary in mesh.boundaries.items():
    fbasis = FacetBasis(mesh, elements, facets=boundary)
    current[port] = asm(port_flux, fbasis, u=fbasis.interpolate(u))

if __name__ == '__main__':

    from skfem.visuals.matplotlib import plot, show

    print('L2 error:', error_L2)
    print('conductance:', conductance)
    print('Current in through ports:', current)

    plot(basis, u)
    show()
