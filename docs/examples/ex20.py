from skfem import *
from skfem.io import from_meshio
from skfem.models.poisson import unit_load

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry


geom = Geometry()
circle = geom.add_circle([0.] * 3, 1., .5**3)
geom.add_physical(circle.line_loop.lines, 'perimeter')
geom.add_physical(circle.plane_surface, 'disk')
mesh = from_meshio(generate_mesh(geom, dim=2))

element = ElementTriMorley()
mapping = MappingAffine(mesh)
ib = InteriorBasis(mesh, element, mapping, 2)


@BilinearForm
def biharmonic(u, v, w):
    from skfem.helpers import ddot, dd

    return ddot(dd(u), dd(v))

stokes = asm(biharmonic, ib)
rotf = asm(unit_load, ib)

psi = solve(*condense(stokes, rotf, D=ib.find_dofs()))
psi0, = ib.interpolator(psi)(np.zeros((2, 1)))

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import draw
    from matplotlib.tri import Triangulation

    print('psi0 = {} (cf. exact = 1/64 = {})'.format(psi0, 1/64))

    M, Psi = ib.refinterp(psi, 3)

    ax = draw(mesh)
    ax.tricontour(Triangulation(*M.p, M.t.T), Psi)
    name = splitext(argv[0])[0]
    ax.get_figure().savefig(f'{name}_stream-lines.png')

    refbasis = InteriorBasis(M, ElementTriP1())
    velocity = np.vstack([derivative(Psi, refbasis, refbasis, 1),
                          -derivative(Psi, refbasis, refbasis, 0)])
    ax = draw(mesh)
    sparsity_factor = 2**3      # subsample the arrows
    vector_factor = 2**3        # lengthen the arrows
    x = M.p[:, ::sparsity_factor]
    u = vector_factor * velocity[:, ::sparsity_factor]
    ax.quiver(*x, *u, x[0])
    ax.get_figure().savefig(f'{name}_velocity-vectors.png')
