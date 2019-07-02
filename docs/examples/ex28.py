from pathlib import Path

from skfem import *
from skfem.importers import from_meshio
from skfem.models.poisson import laplace

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
points = []
lines = []

height = 0.1
length = 1.
thickness = height

lcar = height / 2**3

for xy in [(0., height / 2),
           (0., -height / 2),
           (length, -height / 2),
           (length, height / 2),
           (0., -height / 2 - thickness),
           (length, -height / 2 - thickness)]:
    points.append(geom.add_point([*xy, 0.], lcar))

lines.append(geom.add_line(*points[:2]))
geom.add_physical(lines[-1], 'inlet')

lines.append(geom.add_line(*points[1:3]))

lines.append(geom.add_line(*points[2:4]))
lines.append(geom.add_line(points[3], points[0]))

geom.add_physical(geom.add_plane_surface(geom.add_line_loop(lines)), 'fluid')

lines.append(geom.add_line(points[1], points[4]))

lines.append(geom.add_line(*points[4:6]))
geom.add_physical(lines[-1], 'heated')

lines.append(geom.add_line(points[5], points[2]))

geom.add_physical(geom.add_plane_surface(geom.add_line_loop(
    [*lines[-3:], -lines[1]])), 'wall')


mesh = from_meshio(generate_mesh(geom, dim=2))

elements = ElementTriP1()
basis = InteriorBasis(mesh, elements)

A = asm(laplace, basis)
D = basis.get_dofs(mesh.boundaries)
I = basis.complement_dofs(D)

temperature = np.zeros(basis.N)
temperature[D['heated'].all()] = 1.
temperature[I] = solve(*condense(
    A, np.zeros_like(temperature), temperature, I=I))

mesh.save(Path(__file__).with_suffix('.msh').name,
          point_data={'temperature': temperature})
