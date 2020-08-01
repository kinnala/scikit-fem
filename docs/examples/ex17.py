r"""Insulated wire.

.. note::
   This example requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`_.

This example solves the steady heat conduction
with generation in an insulated wire. In radial
coordinates, the governing equations read: find :math:`T`
satisfying

.. math::
   \nabla \cdot (k_0 \nabla T) + A = 0, \quad 0<r<a,
and

.. math::
   \nabla \cdot (k_1 \nabla T) = 0, \quad a<r<b,
with the boundary condition

.. math::
   k_1 \frac{\partial T}{\partial r} + h T = 0, \quad \text{on $r=b$}.
The parameter values are :math:`k_0 = 101`, :math:`k_1 = 11`, :math:`A = 5`,
:math:`h = 7`, and the geometry is defined as :math:`a=2` and :math:`b=3`.

For comparison purposes, the exact solution at the origin is

.. math::
   T(r=0) = \frac{A b^2}{4 k_0} \left( \frac{2k_0}{bh} + \frac{2 k_0}{k_1} \log \frac{b}{a} + 1\right).

License
-------

Copyright 2018-2020 scikit-fem developers

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
from typing import Optional

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import *
from skfem.helpers import dot, grad
from skfem.models.poisson import mass, unit_load
from skfem.io import from_meshio

radii = [2., 3.]
joule_heating = 5.
heat_transfer_coefficient = 7.
thermal_conductivity = {'wire': 101.,  'insulation': 11.}


def make_mesh(a: float,         # radius of wire
              b: float,         # radius of insulation
              dx: Optional[float] = None) -> MeshTri:

    dx = a / 2 ** 3 if dx is None else dx

    origin = np.zeros(3)
    geom = Geometry()
    wire = geom.add_circle(origin, a, dx, make_surface=True)
    geom.add_physical(wire.plane_surface, 'wire')
    insulation = geom.add_circle(origin, b, dx, holes=[wire.line_loop])
    geom.add_physical(insulation.plane_surface, 'insulation')
    geom.add_physical(insulation.line_loop.lines, 'convection')
    geom.add_raw_code('Mesh.RecombineAll=1;')
    geom.add_raw_code('Mesh.RecombinationAlgorithm=2;\n')

    return from_meshio(generate_mesh(geom, dim=2))


mesh = make_mesh(*radii)


@BilinearForm
def conduction(u, v, w):
    return dot(w['conductivity'] * grad(u), grad(v))


convection = mass

element = ElementQuad1()
basis = InteriorBasis(mesh, element)

conductivity = basis.zero_w()
for subdomain, elements in mesh.subdomains.items():
    conductivity[elements] = thermal_conductivity[subdomain]

L = asm(conduction, basis, conductivity=conductivity)

facet_basis = FacetBasis(mesh, element, facets=mesh.boundaries['convection'])
H = heat_transfer_coefficient * asm(convection, facet_basis)

wire_basis = InteriorBasis(mesh, basis.elem, elements=mesh.subdomains['wire'])
f = joule_heating * asm(unit_load, wire_basis)

temperature = solve(L + H, f)

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import draw, plot, savefig

    T0 = {'skfem': basis.interpolator(temperature)(np.zeros((2, 1)))[0],
          'exact':
          (joule_heating * radii[0]**2 / 4 / thermal_conductivity['wire'] *
           (2 * thermal_conductivity['wire'] / radii[1]
            / heat_transfer_coefficient
            + (2 * thermal_conductivity['wire']
               / thermal_conductivity['insulation']
               * np.log(radii[1] / radii[0])) + 1))}
    print('Central temperature:', T0)

    ax = draw(mesh)
    plot(basis, temperature, ax=ax, colorbar=True)
    savefig(splitext(argv[0])[0] + '_solution.png')
