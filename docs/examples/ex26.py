r"""Restricting a problem to a subdomain.

The `ex17.py` example solved the steady-state heat equation with uniform
volumetric heating in a central core surrounded by an annular insulating layer
of lower thermal conductivity.  Here, the problem is completely restricted to
the core, taking the temperature as zero throughout the annulus.

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

"""
from skfem import *
from skfem.models.poisson import laplace, unit_load

import numpy as np

from .ex17 import mesh, basis, radii,\
    joule_heating, thermal_conductivity


annulus = np.unique(basis.element_dofs[:, mesh.subdomains['annulus']])
temperature = basis.zeros()
core = basis.complement_dofs(annulus)
core_basis = InteriorBasis(mesh, basis.elem, elements=mesh.subdomains['core'])
L = asm(laplace, core_basis)
f = asm(unit_load, core_basis)
temperature = solve(*condense(thermal_conductivity['core'] * L,
                              joule_heating * f,
                              D=annulus))

if __name__ == '__main__':
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import draw, plot

    T0 = {'skfem': basis.probes(np.zeros((2, 1)))(temperature)[0],
          'exact':
          joule_heating * radii[0]**2 / 4 / thermal_conductivity['core']}
    print('Central temperature:', T0)

    ax = draw(mesh)
    plot(mesh, temperature[basis.nodal_dofs.flatten()],
         ax=ax, edgecolors='none', colorbar=True)
    ax.get_figure().savefig(splitext(argv[0])[0] + '_solution.png')
