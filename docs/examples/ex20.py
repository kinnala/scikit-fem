from skfem import *
from skfem.io import from_meshio

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


@bilinear_form
def biharmonic(u, du, ddu, v, dv, ddv, w):

    def shear(ddw):
        return np.array([[ddw[0][0], ddw[0][1]],
                         [ddw[1][0], ddw[1][1]]])

    def ddot(T1, T2):
        return T1[0, 0]*T2[0, 0] +\
               T1[0, 1]*T2[0, 1] +\
               T1[1, 0]*T2[1, 0] +\
               T1[1, 1]*T2[1, 1]

    return ddot(shear(ddu), shear(ddv))


@linear_form
def unit_rotation(v, dv, ddv, w):
    return v


stokes = asm(biharmonic, ib)
rotf = asm(unit_rotation, ib)

dofs = ib.get_dofs(mesh.boundaries['perimeter'])

D = np.concatenate((dofs.nodal['u'], dofs.facet['u_n']))

psi = solve(*condense(stokes, rotf, D=D))


from matplotlib.tri import Triangulation

# Evaluate the stream-function at the origin.
psi0, = ib.interpolator(psi)(np.zeros((2, 1)))
    
if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import draw

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
