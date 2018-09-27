"""The stream-function ψ for two-dimensional incompressible creeping flow 

is governed by the biharmonic equation, ν Δ²ψ = rot f, where ν is the
kinematic viscosity, f the volumetric body-force, and rot f = (∂f/∂y,
-∂f/∂x).  The boundary conditions at a wall are that ψ be constant
(the wall is impermeable) and that the normal component of its
gradient vanish (no slip).  Thus the boundary value problem is
analogous to that of bending a plate, and may be treated with Morley
elements as in ex10.

Here consider a buoyancy force f = (0, x), which arises in the
Boussinesq approximation of natural convection with a horizontal
temperature gradient.

For a circular cavity, the problem admits a polynomial solution with
circular stream-lines.

"""

from skfem import *

from matplotlib.tri import Triangulation
import numpy as np

import meshio
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
circle = geom.add_circle([0.] * 3, 1., .5**3)
geom.add_physical_line(circle.line_loop.lines, 'perimeter')
geom.add_physical_surface(circle.plane_surface, 'disk')
mesh = MeshTri.from_meshio(meshio.Mesh(*generate_mesh(geom)))

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
rotf = -asm(unit_rotation, ib)

dofs = ib.get_dofs(mesh.boundaries['perimeter'])

D = np.concatenate((dofs.nodal['u'], dofs.facet['u_n']))

psi = np.zeros_like(rotf)
psi[ib.complement_dofs(D)] = solve(*condense(stokes, rotf, D=D))

if __name__ == "__main__":

    from os.path import splitext
    from sys import argv

    M, Psi = ib.refinterp(psi, 3)
    ax = mesh.draw()
    ax.tricontour(Triangulation(M.p[0, :], M.p[1, :], M.t.T), Psi)
    ax.axis('off')
    ax.get_figure().savefig(splitext(argv[0])[0] + '.png')
