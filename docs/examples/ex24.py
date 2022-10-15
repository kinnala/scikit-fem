r"""Backward-facing step.

Following the example :ref:`stokesex`, this is another example of the Stokes flow.  The
difference here is that the domain has an inlet (with an imposed velocity) and
an outlet (through which fluid issues against a uniform pressure).

The geometry is taken from Barkley et al (2002); i.e. an expansion ratio of 2,
one step-length upstream and 35 downstream.

* Barkley, D., M. G. M. Gomes, & R. D. Henderson (2002). Three-dimensional instability in flow over a backward-facing step. Journal of Fluid Mechanics 473. pp. 167–190. `doi:10.1017/s002211200200232x <http://dx.doi.org/10.1017/s002211200200232x>`_

"""

from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np

from skfem import *
from skfem.models.poisson import vector_laplace, laplace
from skfem.models.general import divergence, rot
from skfem.io.json import from_file

mesh = from_file(Path(__file__).parent / 'meshes' / 'backward-facing_step.json')

element = {'u': ElementVector(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: Basis(mesh, e, intorder=3)
         for variable, e in element.items()}

D = basis['u'].get_dofs(['inlet', 'ceiling', 'floor'])

A = asm(vector_laplace, basis['u'])
B = -asm(divergence, basis['u'], basis['p'])

K = bmat([[A, B.T],
          [B, None]], 'csr')

inlet_basis = FacetBasis(mesh, element['u'], facets=mesh.boundaries['inlet'])


def parabolic(x):
    """return the plane Poiseuille parabolic inlet profile"""
    return np.stack([4 * x[1] * (1. - x[1]), np.zeros_like(x[0])])


uvp = np.hstack((
    inlet_basis.project(parabolic),
    basis['p'].zeros(),
))
uvp = solve(*condense(K, x=uvp, D=D))

velocity, pressure = np.split(uvp, K.blocks)

basis['psi'] = basis['u'].with_element(ElementTriP2())
A = asm(laplace, basis['psi'])
psi = basis['psi'].zeros()
vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
psi = solve(*condense(A, vorticity, D=basis['psi'].get_dofs('floor')))


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
    contour = partial(ax.tricontour,
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
        contour(levels=levels, colors=color, linestyles=style)

    ax.set_aspect(1.)
    ax.axis('off')
    fig.savefig(f'{name}-stream-function.png',
                bbox_inches='tight', pad_inches=0)
