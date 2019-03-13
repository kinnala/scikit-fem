from itertools import cycle, islice

from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse import bmat

import meshio
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import *
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

inlet_basis = FacetBasis(mesh, element['u'], facets=mesh.boundaries['inlet'])
inlet_dofs_ = inlet_basis.get_dofs(mesh.boundaries['inlet'])
inlet_dofs = np.concatenate([inlet_dofs_.nodal[f'u^{1}'],
                             inlet_dofs_.facet[f'u^{1}']])


def parabolic(x, y):
    """return the plane Poiseuille parabolic inlet profile"""
    return np.moveaxis(np.dstack([4 * y * (1. - y), np.zeros_like(y), ]),
                       [0, 1, 2], [1, 2, 0])


uvp[inlet_dofs] = L2_projection(parabolic, inlet_basis, inlet_dofs)
I = np.setdiff1d(np.arange(K.shape[0]), D)
uvp[I] = solve(*condense(K, 0*uvp, uvp, I))

velocity, pressure = np.split(uvp, [A.shape[0]])

basis['psi'] = InteriorBasis(mesh, ElementTriP2())
A = asm(laplace, basis['psi'])
psi = np.zeros(basis['psi'].N)
D = basis['psi'].get_dofs(mesh.boundaries['floor']).all()
I = basis['psi'].complement_dofs(D)
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi[I] = solve(*condense(A, vorticity, I=I))


if __name__ == '__main__':

    from functools import partial
    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    name = splitext(argv[0])[0]
    
    ax = mesh.plot(pressure)
    ax.get_figure().savefig(f'{name}-pressure.png',
                            bbox_inches='tight', pad_inches=0)

    mesh.save(f'{name}-velocity.vtk', velocity[basis['u'].nodal_dofs].T)
    
    fig, ax = subplots()
    ax.plot(
        *mesh.p[:, mesh.facets[:, np.concatenate(list(mesh.boundaries.values()))]],
        color='k')

    n_streamlines = 11
    plot = partial(ax.tricontour,
                   Triangulation(mesh.p[0, :], mesh.p[1, :], mesh.t.T),
                   psi[basis['psi'].nodal_dofs.flatten()],
                   linewidths=1.)
    for levels, color, style in [
            (np.linspace(0, 2/3, n_streamlines),
             'k',
             ['dashed'] + ['solid']*(n_streamlines - 2) + ['dashed']),
            (np.linspace(2/3, max(psi), n_streamlines)[0:],
             'r', 'solid'),
            (np.linspace(min(psi), 0, n_streamlines)[:-1],
             'g', 'solid')]:
        plot(levels=levels, colors=color, linestyles=style)

    ax.set_aspect(1.)
    ax.axis('off')
    fig.savefig(f'{name}-stream-function.png',
                bbox_inches='tight', pad_inches=0)
