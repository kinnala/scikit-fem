from skfem import *
from skfem.models.poisson import laplace, mass

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
geom.add_physical(lines[-1], 'ground')
lines.append(geom.add_circle_arc(points[2], points[0], points[3]))
lines.append(geom.add_line(points[3], points[4]))
geom.add_physical(lines[-1], 'positive')
lines.append(geom.add_circle_arc(points[4], points[0], points[1]))
geom.add_physical(
    geom.add_plane_surface(geom.add_line_loop(lines)), 'domain')

mesh = MeshTri.from_meshio(generate_mesh(geom, dim=2))

elements = ElementTriP2()
basis = InteriorBasis(mesh, elements)
A = asm(laplace, basis)

boundary_dofs = basis.get_dofs(mesh.boundaries)
interior_dofs = basis.complement_dofs(boundary_dofs)

u = np.zeros(basis.N)
u[boundary_dofs['positive'].all()] = 1.
u[interior_dofs] = solve(*condense(A, 0.*u, u, interior_dofs))

M = asm(mass, basis)
u_exact = L2_projection(lambda x, y: 2 * np.arctan2(y, x) / np.pi, basis)
u_error = u - u_exact
print('L2 error =', np.sqrt(u_error @ M @ u_error))
print('conductance = {:.4f} (exact = 2 ln 2 / pi = {:.4f})'.format(
    u @ A @ u, 2 * np.log(2) / np.pi))

@linear_form
def port_flux(v, dv, w):
    return sum(w.n * dv)

for port in mesh.boundaries:
    basis = FacetBasis(mesh, elements, facets=mesh.boundaries[port])
    form = asm(port_flux, basis)
    print('Current in through {} = {:.4f}'.format(port, form @ u))

mesh.plot(u[basis.nodal_dofs.flatten()])
mesh.show()
