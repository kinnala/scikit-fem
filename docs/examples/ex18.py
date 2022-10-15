r"""Stokes equations.

This solves for the creeping flow problem in the primitive variables,
i.e. velocity and pressure instead of the stream-function.  These are governed
by the Stokes momentum

.. math::
    0 = -\rho^{-1}\nabla p + \boldsymbol{f} + \nu\Delta\boldsymbol{u}
and continuity equations

.. math::
    \nabla\cdot\boldsymbol{u} = 0.
This is an example of a mixed problem because it contains two
different kinds of unknowns; pairs of elements for them have to be
chosen carefully.  One of the simplest workable choices is the
Taylor--Hood element: :math:`P_2` for velocity
and :math:`P_1` for pressure.

Once the velocity has been found, the stream-function :math:`\psi` can
be calculated by solving the Poisson problem

.. math::
    -\Delta\psi = \mathrm{rot}\,\boldsymbol{u},
where :math:`\mathrm{rot}\,\boldsymbol{u} \equiv
\partial u_y/\partial x - \partial u_x/\partial y`.
The boundary conditions are that the stream-function is constant
around the impermeable perimeter; this constant can be taken as zero
without loss of generality.  In the weak formulation

.. math::
    \left(\nabla\phi, \nabla\psi\right) = \left(\phi, \mathrm{rot}\,\boldsymbol{u}\right) \quad \forall \phi \in H^1_0(\Omega),
the right-hand side can be converted using Green's theorem and the
no-slip condition to not involve the derivatives of the velocity:

.. math::
     \left(\phi, \mathrm{rot}\,\boldsymbol{u}\right) = \left(\boldsymbol{rot}\,\phi, \boldsymbol{u}\right)
where :math:`\boldsymbol{rot}` is the adjoint of :math:`\mathrm{rot}`:

.. math::
    \boldsymbol{rot}\,\phi \equiv \frac{\partial\phi}{\partial y}\hat{i} - \frac{\partial\phi}{\partial x}\hat{j}.

"""
from skfem import *
from skfem.io.json import from_file
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot

from pathlib import Path

import numpy as np


mesh = MeshTri.init_circle(4)

element = {'u': ElementVector(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: Basis(mesh, e, intorder=3)
         for variable, e in element.items()}


@LinearForm
def body_force(v, w):
    return w.x[0] * v[1]


A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
C = asm(mass, basis['p'])

K = bmat([[A, -B.T],
          [-B, 1e-6 * C]])

f = np.concatenate([asm(body_force, basis['u']),
                    basis['p'].zeros()])

uvp = solve(*condense(K, f, D=basis['u'].get_dofs()))

velocity, pressure = np.split(uvp, K.blocks)

basis['psi'] = basis['u'].with_element(ElementTriP2())
A = asm(laplace, basis['psi'])
vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
psi = solve(*condense(A, vorticity, D=basis['psi'].get_dofs()))


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    from skfem.visuals.matplotlib import plot, draw, savefig

    name = splitext(argv[0])[0]

    mesh.save(f'{name}_velocity.vtk',
              {'velocity': velocity[basis['u'].nodal_dofs].T})


    ax = draw(mesh)
    plot(basis['p'], pressure, ax=ax)
    savefig(f'{name}_pressure.png')

    ax = draw(mesh)
    velocity1 = velocity[basis['u'].nodal_dofs]
    ax.quiver(*mesh.p, *velocity1, mesh.p[0])  # colour by buoyancy
    savefig(f'{name}_velocity.png')

    ax = draw(mesh)
    ax.tricontour(Triangulation(*mesh.p, mesh.t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])
    savefig(f'{name}_stream-function.png')
