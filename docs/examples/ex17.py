r"""Insulated wire.

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


"""
from pathlib import Path
from typing import Optional

import numpy as np

from skfem import *
from skfem.helpers import dot, grad
from skfem.models.poisson import mass, unit_load
from skfem.io.json import from_file

joule_heating = 5.
heat_transfer_coefficient = 7.
thermal_conductivity = {'core': 101.,  'annulus': 11.}

mesh = from_file(Path(__file__).parent / 'meshes' / 'disk.json')
radii = sorted([np.linalg.norm(mesh.p[:, mesh.t[:, s]], axis=0).max() for s in mesh.subdomains.values()])


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

facet_basis = FacetBasis(mesh, element, facets=mesh.boundaries['perimeter'])
H = heat_transfer_coefficient * asm(convection, facet_basis)

core_basis = InteriorBasis(mesh, basis.elem, elements=mesh.subdomains['core'])
f = joule_heating * asm(unit_load, core_basis)

temperature = solve(L + H, f)

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import draw, plot, savefig

    T0 = {'skfem': (basis.probes(np.zeros((2, 1))) @ temperature)[0],
          'exact':
          (joule_heating * radii[0]**2 / 4 / thermal_conductivity['core'] *
           (2 * thermal_conductivity['core'] / radii[1]
            / heat_transfer_coefficient
            + (2 * thermal_conductivity['core']
               / thermal_conductivity['annulus']
               * np.log(radii[1] / radii[0])) + 1))}
    print('Central temperature:', T0)

    ax = draw(mesh)
    plot(basis, temperature, ax=ax, colorbar=True)
    savefig(splitext(argv[0])[0] + '_solution.png')
