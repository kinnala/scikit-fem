"""This solves for the same creeping flow as ex18 in primitive variables,

i.e. velocity and pressure instead of stream-function.  These are governed by
the Stokes momentum

.. math::
    0 = -\rho^{-1}\nabla p + \mathbf f \nu\Delta\mathbf u

and continuity equations

.. math::
    \nabla\mathbf u = 0.

This is an example of a mixed problem because it contains two
different kinds of unknowns; pairs of elements for them have to be
chosen carefully.  One of the simplest workable choices is the
Taylor--Hood element: `ElementVectorH1(ElementTriP2())` for velocity
and `ElementTriP1()` for pressure.

This example also demonstrates the use of the external pure-Python
`dmsh` to generate a `MeshTri`.

Once the velocity has been found, the stream-function :math:`\psi` can
be calculated by solving the Poisson problem

.. math::
    -\Delta\psi = \mathbf k\cdot\nabla\times\mathbf u

where :math:`\mathbf k` is the unit-vector out of the plane.  The
boundary conditions are that the stream-function be constant around
the impermeable perimeter; this constant can be taken as zero without
loss of generality.  In the weak formulation

.. math::
    \left(\nabla\phi, \nabla\psi\right) = \left(\phi\mathbf k, \nabla\times\mathbf u\right)

the right-hand side can be converted using Green's theorem and the
no-slip condition to not involve the derivatives of the velocity:

.. math::
     \left(\phi\mathbf k, \nabla\times\mathbf u\right) = \left(\mathbf k\times\nabla\phi, \mathbf u\right).

"""


from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence

from matplotlib.tri import Triangulation
import numpy as np
from scipy.sparse import bmat

import dmsh

mesh = MeshTri(*map(np.transpose,
                    dmsh.generate(dmsh.Circle([0., 0.], 1.), .1)))

element = {'u': ElementVectorH1(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@linear_form
def body_force(v, dv, w):
    return w.x[0] * v[1]


A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
C = asm(mass, basis['p'])

K = bmat([[A, B.T],
          [B, 1e-3 * C]]).tocsr()

f = np.concatenate([asm(body_force, basis['u']),
                    np.zeros(B.shape[0])])

boundary = mesh.submesh(boundaries_only=True)
dofs = basis['u'].get_dofs(boundary)
D = np.concatenate((dofs.nodal['u^1'], dofs.nodal['u^2']))
uvp = np.zeros(K.shape[0])
uvp[np.setdiff1d(np.arange(K.shape[0]), D)] = solve(*condense(K, f, D=D))

velocity, pressure = np.split(uvp, [A.shape[0]])

ax = mesh.plot(pressure)
ax.axis('off')
ax.get_figure().savefig('taylor_hood_pressure.png')

ax = mesh.draw()
velocity1 = velocity[basis['u'].nodal_dofs]
ax.quiver(mesh.p[0, :], mesh.p[1, :],
          velocity1[0, :], velocity1[1, :],
          mesh.p[0, :])         # colour by buoyancy
ax.axis('off')
ax.get_figure().savefig('taylor_hood_velocity.png')


@linear_form
def rot(v, dv, w):
    return dv[1] * w.w[0] - dv[0] * w.w[1]


basis['psi'] = InteriorBasis(mesh, ElementTriP2())
A = asm(laplace, basis['psi'])
psi = np.zeros(A.shape[0])
D = basis['psi'].get_dofs(boundary).nodal['u']
interior = basis['psi'].complement_dofs(D)
psi[D] = 0.
vorticity = asm(rot, basis['psi'],
                w=(basis['psi'].interpolate(velocity[::2]),
                   basis['psi'].interpolate(velocity[1::2])))
psi[interior] = solve(*condense(A, vorticity, I=interior))

ax = mesh.draw()
ax.tricontour(Triangulation(mesh.p[0, :], mesh.p[1, :], mesh.t.T),
              psi[basis['psi'].nodal_dofs.flatten()])
ax.axis('off')
ax.get_figure().savefig('taylor_hood_stream-function.png')
