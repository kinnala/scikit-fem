from typing import Optional

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import *
from skfem.models.poisson import laplace, unit_load


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

    return MeshTri.from_meshio(generate_mesh(geom, dim=2))


radii = [2., 3.]
joule_heating = 5.
thermal_conductivity = {'wire': 101.}

mesh = make_mesh(*radii)
element = ElementTriP2()
basis = InteriorBasis(mesh, element)

L = asm(laplace, basis)
f = asm(unit_load, basis)

insulation = np.unique(basis.element_dofs[:, mesh.subdomains['insulation']])
temperature = np.zeros(basis.N)
wire = basis.complement_dofs(insulation)
temperature[wire] = solve(*condense(thermal_conductivity['wire'] * L,
                                    joule_heating * f,
                                    D=insulation))

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    T0 = {'skfem': basis.interpolator(temperature)(np.zeros((2, 1)))[0],
          'exact':
          joule_heating * radii[0]**2 / 4 / thermal_conductivity['wire']}
    print('Central temperature:', T0)

    mesh.plot(temperature[basis.nodal_dofs.flatten()], colorbar=True)
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
