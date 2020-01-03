from typing import Optional

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import *
from skfem.models.poisson import mass
from skfem.importers.meshio import from_meshio

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


@bilinear_form
def conduction(u, du, v, dv, w):
    return w.w * sum(du * dv)


convection = mass

element = ElementQuad1()
basis = InteriorBasis(mesh, element)

conductivity = basis.zero_w()
for subdomain, elements in mesh.subdomains.items():
    conductivity[elements] = thermal_conductivity[subdomain]

L = asm(conduction, basis, w=conductivity)

facet_basis = FacetBasis(mesh, element, facets=mesh.boundaries['convection'])
H = heat_transfer_coefficient * asm(convection, facet_basis)


@linear_form
def generation(v, dv, w):
    return w.w * v


heated = basis.zero_w()
heated[mesh.subdomains['wire']] = 1.
f = joule_heating * asm(generation, basis, w=heated)

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
