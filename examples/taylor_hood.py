from skfem import *
from skfem.models.poisson import mass

import numpy as np
from scipy.sparse import bmat

import meshio
from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
circle = geom.add_circle([0.] * 3, 1., .5**3)
geom.add_physical_line(circle.line_loop.lines, 'perimeter')
geom.add_physical_surface(circle.plane_surface, 'disk')
mesh = MeshTri.from_meshio(meshio.Mesh(*generate_mesh(geom)))

element = {'u': ElementVectorH1(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@bilinear_form
def vector_laplacian(u, du, v, dv, w):
    return np.einsum('ij...,ij...', du, dv)


@bilinear_form
def dilatation(u, du, q, dq, w):
    return (du[0][0] + du[1][1]) * q


@linear_form
def body_force(v, dv, w):
    return w.x[0] * v[1]


A = asm(vector_laplacian, basis['u'])
B = asm(dilatation, basis['u'], basis['p'])
C = asm(mass, basis['p'])

K = bmat([[A, B.T],
          [B, 1e-3 * C]]).tocsr()

f = np.concatenate([asm(body_force, basis['u']),
                    np.zeros(B.shape[0])])

dofs = basis['u'].get_dofs(mesh.boundaries['perimeter'])
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
