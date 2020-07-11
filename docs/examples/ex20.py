r"""Creeping flow.

.. note::
   This example requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`_.

The stream-function :math:`\psi` for two-dimensional creeping flow is
governed by the biharmonic equation

.. math::  
    \nu \Delta^2\psi = \mathrm{rot}\,\boldsymbol{f}
where :math:`\nu` is the kinematic viscosity (assumed constant),
:math:`\boldsymbol{f}` the volumetric body-force, and :math:`\mathrm{rot}\,\boldsymbol{f} \equiv
\partial f_y/\partial x - \partial f_x/\partial y`.  The boundary
conditions at a wall are that :math:`\psi` is constant (the wall is
impermeable) and that the normal component of its gradient vanishes (no
slip).  Thus, the boundary value problem is analogous to that of
bending a clamped plate, and may be treated with Morley elements as in
the Kirchhoff plate tutorial.

Here we consider a buoyancy force :math:`\boldsymbol{f} = x\hat{j}`,
which arises in the Boussinesq approximation of natural convection
with a horizontal temperature gradient (`Batchelor 1954
<http://dx.doi.org/10.1090/qam/64563>`_).

For a circular cavity of radius :math:`a`, the problem admits a
polynomial solution with circular stream-lines:

.. math::
    \psi = \left(1 - (x^2+y^2)/a^2\right)^2 / 64.

"""
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
