from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot

import numpy as np
from scipy.sparse import bmat

from time import time

mesh = MeshTri.init_tensor(*(np.linspace(0., 1., 2**8),)*2)

element = {'u': ElementVectorH1(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@linear_form
def body_force(v, dv, w):
    return w.x[0] * v[1]

tic = time()
A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
C = asm(mass, basis['p'])

K = bmat([[A, -B.T],
          [-B, 1e-6 * C]]).tocsr()

f = np.concatenate([asm(body_force, basis['u']),
                    np.zeros(B.shape[0])])

D = basis['u'].get_dofs().all()
uvp = np.zeros(K.shape[0])
uvp[np.setdiff1d(np.arange(K.shape[0]), D)] = solve(*condense(K, f, D=D))

velocity, pressure = np.split(uvp, [A.shape[0]])
print('elapsed time:', time() - tic)


@linear_form
def rot(v, dv, w):
    return dv[1] * w.w[0] - dv[0] * w.w[1]

basis['psi'] = InteriorBasis(mesh, ElementTriP2())
A = asm(laplace, basis['psi'])
psi = np.zeros(A.shape[0])
D = basis['psi'].get_dofs().all()
interior = basis['psi'].complement_dofs(D)
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi[interior] = solve(*condense(A, vorticity, I=interior))


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    name = splitext(argv[0])[0]

    mesh.save(f'{name}_velocity.vtk', velocity[basis['u'].nodal_dofs].T)
    
    print(basis['psi'].interpolator(psi)(np.zeros((2, 1)))[0],
          '(cf. exact 1/64)')

    print(basis['p'].interpolator(pressure)(np.array([[-0.5, 0.5],
                                                      [0.5, 0.5]])),
          '(cf. exact -/+ 1/8)')

    ax = mesh.plot(pressure)
    ax.get_figure().savefig(f'{name}_pressure.png')

    ax = mesh.draw()
    velocity1 = velocity[basis['u'].nodal_dofs]
    ax.quiver(mesh.p[0, :], mesh.p[1, :],
              velocity1[0, :], velocity1[1, :],
              mesh.p[0, :])         # colour by buoyancy
    ax.get_figure().savefig(f'{name}_velocity.png')

    ax = mesh.draw()
    ax.tricontour(Triangulation(mesh.p[0, :], mesh.p[1, :], mesh.t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])
    ax.get_figure().savefig(f'{name}_stream-function.png')
