from itertools import cycle, islice

import numpy as np

import meshio
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

import skfem
from skfem.models.poisson import laplace


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


def make_mesh(*args, **kwargs) -> skfem.MeshTri:
    return skfem.MeshTri.from_meshio(meshio.Mesh(*generate_mesh(
        make_geom(*args, **kwargs))))


mesh = make_mesh(lcar=.5**2)

element = {'p': skfem.ElementTriP1()}
basis = {variable: skfem.InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}

L = skfem.asm(laplace, basis['p'])

boundary_dofs = basis['p'].get_dofs(mesh.boundaries)

p = np.zeros(basis['p'].N)
p[boundary_dofs['inlet'].all()] = 1.
interior_dofs = basis['p'].complement_dofs(
    np.concatenate([boundary_dofs['inlet'].all(),
                    boundary_dofs['outlet'].all()]))
p[interior_dofs] = skfem.solve(*skfem.condense(L, 0*p, p, I=interior_dofs))

mesh.plot(p)
mesh.show()
