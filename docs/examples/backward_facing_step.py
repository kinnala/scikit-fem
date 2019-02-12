"""Flow over a backward-facing step

The geometry is taken from `Barkley et al (2002); i.e. an expansion
ratio of 2, one step-length upstream and 35 downstream.

* Barkley, D., M. G. M. Gomes, & R. D. Henderson (2002). Three-dimensional instability in flow over a backward-facing step. *Journal of Fluid Mechanics* **473**:167–190. `doi:10.1017/s002211200200232x <http://dx.doi.org/10.1017/s002211200200232x>`_

"""

from itertools import cycle, islice

from matplotlib.pyplot import subplots
from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import bmat

import meshio
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import (MeshTri, ElementVectorH1, ElementTriP2, ElementTriP1,
                   InteriorBasis, asm, condense, solve)
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot


def make_geom(length: float = 35.,
              lcar: float = 1.) -> Geometry:
    # Barkley et al (2002, figure 3 a - c)
    geom = Geometry()

    points = []
    for point in [[0, -1, 0],
                  [length, -1, 0],
                  [length, 1, 0],
                  [-1, 1, 0],
                  [-1, 0, 0],
                  [0, 0, 0]]:
        points.append(geom.add_point(point, lcar))

    lines = []
    for termini in zip(points,
                       islice(cycle(points), 1, None)):
        lines.append(geom.add_line(*termini))

    for k, label in [([1], 'outlet'),
                     ([2], 'ceiling'),
                     ([3], 'inlet'),
                     ([0, 4, 5], 'floor')]:
        geom.add_physical_line(list(np.array(lines)[k]), label)

    geom.add_physical_surface(
        geom.add_plane_surface(geom.add_line_loop(lines)), 'domain')

    return geom


def make_mesh(*args, **kwargs) -> MeshTri:
    return MeshTri.from_meshio(meshio.Mesh(*generate_mesh(
        make_geom(*args, **kwargs))))


mesh = make_mesh(lcar=.5**2)

element = {'u': ElementVectorH1(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}

D = np.setdiff1d(basis['u'].get_dofs().all(),
                 basis['u'].get_dofs(mesh.boundaries['outlet']).all())


A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
C = asm(mass, basis['p'])

K = bmat([[A, -B.T],
          [-B, None]]).tocsr()
uvp = np.zeros(K.shape[0])

xy = np.hstack([mesh.p, mesh.p[:, mesh.facets].mean(axis=1)])
inlet_dofs_ = basis['u'].get_dofs(mesh.boundaries['inlet'])
inlet_dofs = [np.concatenate([inlet_dofs_.nodal[f'u^{i}'],
                              inlet_dofs_.facet[f'u^{i}']])
              for i in [1, 2]]
y = xy.flatten('F')[inlet_dofs[1]]

uvp[inlet_dofs[0]] = 4 * y * (1 - y)
I = np.setdiff1d(np.arange(K.shape[0]), D)
uvp[I] = solve(*condense(K, 0*uvp, uvp, I))

velocity, pressure = np.split(uvp, [A.shape[0]])
mesh.save('velocity.vtk', np.vstack([velocity[basis['u'].nodal_dofs],
                                     np.zeros_like(mesh.p[0])]).T)

ax = mesh.plot(pressure)
ax.get_figure().savefig('pressure.png')

basis['psi'] = InteriorBasis(mesh, ElementTriP2())
A = asm(laplace, basis['psi'])
psi = np.zeros(basis['psi'].N)
D = basis['psi'].get_dofs(mesh.boundaries['floor']).all()
I = basis['psi'].complement_dofs(D)
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi[I] = solve(*condense(A, vorticity, I=I))

fig, ax = subplots()
ax.plot(
    *mesh.p[:, mesh.facets[:, np.concatenate(list(mesh.boundaries.values()))]],
    color='k')
ax.tricontour(Triangulation(mesh.p[0, :], mesh.p[1, :], mesh.t.T),
              psi[basis['psi'].nodal_dofs.flatten()])
ax.set_aspect(1.)
ax.axis('off')
fig.savefig('stream-function.png')
