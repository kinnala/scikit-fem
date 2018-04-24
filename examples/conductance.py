from skfem import *
from skfem.models.poisson import laplace, mass, unit_load

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
points = []
lines = []
radii = [1., 2.]
lcar = .1
points.append(geom.add_point([0.] * 3, lcar))  # centre
for x in radii:
    points.append(geom.add_point([x, 0., 0.], lcar))
for y in reversed(radii):
    points.append(geom.add_point([0., y, 0.], lcar))
lines.append(geom.add_line(*points[1:3]))
geom.add_physical_line(lines[-1], 'ground')
lines.append(geom.add_circle_arc(points[2], points[0], points[3]))
lines.append(geom.add_line(points[3], points[4]))
geom.add_physical_line(lines[-1], 'positive')
lines.append(geom.add_circle_arc(points[4], points[0], points[1]))
geom.add_physical_surface(
    geom.add_plane_surface(geom.add_line_loop(lines)), 'domain')

pts, cells, _, cell_data, field_data = generate_mesh(
    geom, prune_vertices=False)

mesh = MeshTri(pts[:, :2].T, cells['triangle'].T)
boundaries = {bc:
              np.unique(cells['line'][cell_data['line']['gmsh:physical'] ==
                                      field_data[bc][0]])
              for bc in field_data if field_data[bc][1] == 1}

elements = ElementTriP1()
basis = InteriorBasis(mesh, elements, MappingAffine(mesh), 2)
A = asm(laplace, basis)
b = asm(unit_load, basis)

dofs = np.setdiff1d(np.arange(0, mesh.p.shape[1]),
                    np.union1d(boundaries['positive'],
                               boundaries['ground']))

u = 0.*b
u[boundaries['positive']] = 1.
u[dofs] = solve(*condense(A, 0.*b, u, dofs))

u_exact = 2 * np.arctan2(mesh.p[1, :], mesh.p[0, :]) / np.pi
u_error = u - u_exact
print('L2 error =', np.sqrt(u_error @ asm(mass, basis) @ u_error))
print('conductance = {:.4f} (exact = 2 ln 2 / pi = {:.4f})'.format(
    u @ A @ u, 2 * np.log(2) / np.pi))

mesh.plot3(u)
mesh.show()
