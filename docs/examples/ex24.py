from itertools import cycle, islice

from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse import bmat

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot
from skfem.io import from_meshio


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
        geom.add_physical(list(np.array(lines)[k]), label)

    geom.add_physical(
        geom.add_plane_surface(geom.add_line_loop(lines)), 'domain')

    return geom


def make_mesh(*args, **kwargs) -> MeshTri:
    return from_meshio(generate_mesh(make_geom(*args, **kwargs), dim=2))


mesh = make_mesh(lcar=.5**2)

element = {'u': ElementVectorH1(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}

del mesh.boundaries['outlet']
D = np.concatenate([b.all() for b in basis['u'].find_dofs().values()])

A = asm(vector_laplace, basis['u'])
B = -asm(divergence, basis['u'], basis['p'])

K = bmat([[A, B.T],
          [B, None]], 'csr')
uvp = np.zeros(K.shape[0])

inlet_basis = FacetBasis(mesh, element['u'], facets=mesh.boundaries['inlet'])
inlet_dofs = inlet_basis.find_dofs()['inlet'].all()


def parabolic(x, y):
    """return the plane Poiseuille parabolic inlet profile"""
    return 4 * y * (1. - y), np.zeros_like(y)


uvp[inlet_dofs] = L2_projection(parabolic, inlet_basis, inlet_dofs)
uvp = solve(*condense(K, np.zeros_like(uvp), uvp, D=D))

velocity, pressure = np.split(uvp, [A.shape[0]])

basis['psi'] = InteriorBasis(mesh, ElementTriP2())
A = asm(laplace, basis['psi'])
psi = np.zeros(basis['psi'].N)
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi = solve(*condense(A, vorticity, D=basis['psi'].find_dofs()['floor'].all()))


if __name__ == '__main__':

    from functools import partial
    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    from skfem.visuals.matplotlib import plot, savefig

    name = splitext(argv[0])[0]

    plot(mesh, pressure)
    savefig(f'{name}-pressure.png', bbox_inches='tight', pad_inches=0)

    mesh.save(f'{name}-velocity.vtk',
              {'velocity': velocity[basis['u'].nodal_dofs].T})

    fig, ax = subplots()
    ax.plot(*mesh.p[:, mesh.facets[:, np.concatenate(
        list(mesh.boundaries.values()))]],
            color='k')

    n_streamlines = 11
    plot = partial(ax.tricontour,
                   Triangulation(*mesh.p, mesh.t.T),
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
