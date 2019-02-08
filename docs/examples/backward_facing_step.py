"""Flow over a backward-facing step

The geometry is taken from `Barkley et al (2002); i.e. an expansion
ratio of 2, one step-length upstream and 35 downstream.

* Barkley, D., M. G. M. Gomes, & R. D. Henderson (2002). Three-dimensional instability in flow over a backward-facing step. *Journal of Fluid Mechanics* **473**:167–190. `doi:10.1017/s002211200200232x <http://dx.doi.org/10.1017/s002211200200232x>`_

"""

from itertools import cycle, islice

import numpy as np
from scipy.sparse import bmat

import meshio
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import (MeshTri,
                   ElementVectorH1, ElementTriP2, ElementTriP1,
                   InteriorBasis, asm,
                   condense, solve)
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence


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

boundary_dofs = basis['p'].get_dofs(mesh.boundaries)

# impulsive pressure

p = np.zeros(basis['p'].N)
p[boundary_dofs['inlet'].all()] = 1.
interior_dofs = basis['p'].complement_dofs(
    np.concatenate([boundary_dofs['inlet'].all(),
                    boundary_dofs['outlet'].all()]))
L = asm(laplace, basis['p'])
p[interior_dofs] = solve(*condense(L, 0*p, p, I=interior_dofs))

mesh.plot(p).get_figure().savefig('impulsive.png')

# creeping flow

D = np.setdiff1d(basis['u'].get_dofs().all(),
                 basis['u'].get_dofs(mesh.boundaries['outlet']).all())


A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
C = asm(mass, basis['p'])

K = bmat([[A, -B.T],
          [-B, None]]).tocsr()
uvp = np.zeros(K.shape[0])

# TODO: u (y) = 4 y (1 - y) on 'inlet' (0 < y < 1)  #112

# For the moment use the mean
uvp[basis['u'].get_dofs(mesh.boundaries['inlet']).nodal['u^1']] = 2/3
I = np.setdiff1d(np.arange(K.shape[0]), D)
uvp[I] = solve(*condense(K, 0*uvp, uvp, I))

velocity, pressure = np.split(uvp, [A.shape[0]])

ax = mesh.plot(pressure)
ax.get_figure().savefig('pressure.png')

ax = mesh.draw()
velocity1 = velocity[basis['u'].nodal_dofs]
ax.quiver(mesh.p[0, :], mesh.p[1:, ],
          velocity1[0, :], velocity1[1, :])
ax.get_figure().savefig('velocity.png')
