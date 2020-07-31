r"""Restricting a problem to a subdomain.

.. note::
   This example requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`_.

The `ex17.py` example solved the steady-state heat equation with uniform
volumetric heating in a central wire surrounded by an annular insulating layer
of lower thermal conductivity.  Here, the problem is completely restricted to
the wire, taking the temperature as zero throughout the annulus.

Thus the problem reduces to the same Poisson equation with uniform forcing and
homogeneous Dirichlet conditions:

.. math::
   \nabla\cdot(k\nabla T) + A  = 0, \qquad 0 < r < a
with

.. math::
   T = 0, \qquad\text{on}\quad r = a.
The exact solution is

.. math::
   T = \frac{s}{4k}(a^2 - r^2).

The novelty here is that the temperature is defined as a finite element function
throughout the mesh (:math:`r < b`) but only solved on a subdomain.

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
from skfem import *
from skfem.models.poisson import laplace, unit_load

import numpy as np

from docs.examples.ex17 import mesh, basis, radii,\
    joule_heating, thermal_conductivity


insulation = np.unique(basis.element_dofs[:, mesh.subdomains['insulation']])
temperature = np.zeros(basis.N)
wire = basis.complement_dofs(insulation)
wire_basis = InteriorBasis(mesh, basis.elem, elements=mesh.subdomains['wire'])
L = asm(laplace, wire_basis)
f = asm(unit_load, wire_basis)
temperature = solve(*condense(thermal_conductivity['wire'] * L,
                              joule_heating * f,
                              D=insulation))

if __name__ == '__main__':
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import draw, plot

    T0 = {'skfem': basis.interpolator(temperature)(np.zeros((2, 1)))[0],
          'exact':
          joule_heating * radii[0]**2 / 4 / thermal_conductivity['wire']}
    print('Central temperature:', T0)

    ax = draw(mesh)
    plot(mesh, temperature[basis.nodal_dofs.flatten()],
         ax=ax, edgecolors='none', colorbar=True)
    ax.get_figure().savefig(splitext(argv[0])[0] + '_solution.png')
