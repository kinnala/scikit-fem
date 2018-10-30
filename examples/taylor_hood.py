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

"""


from skfem import *
from skfem.models.poisson import vector_laplace, mass
from skfem.models.general import divergence

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

dofs = basis['u'].get_dofs(mesh.submesh(boundaries_only=True))
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
