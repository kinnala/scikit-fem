from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
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

K = bmat([[A, -B.T],
          [-B, 1e-6 * C]]).tocsr()

f = np.concatenate([asm(body_force, basis['u']),
                    np.zeros(B.shape[0])])

boundary = mesh.submesh(boundaries_only=True)
dofs = basis['u'].get_dofs(boundary)
D = dofs.all()
uvp = np.zeros(K.shape[0])
uvp[np.setdiff1d(np.arange(K.shape[0]), D)] = solve(*condense(K, f, D=D))

velocity, pressure = np.split(uvp, [A.shape[0]])

@linear_form
def rot(v, dv, w):
    return dv[1] * w.w[0] - dv[0] * w.w[1]


basis['psi'] = InteriorBasis(mesh, ElementTriP2())
A = asm(laplace, basis['psi'])
psi = np.zeros(A.shape[0])
D = basis['psi'].get_dofs(boundary).all()
interior = basis['psi'].complement_dofs(D)
psi[D] = 0.
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi[interior] = solve(*condense(A, vorticity, I=interior))


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    name = splitext(argv[0])[0]
    
    print(basis['p'].interpolator(pressure)(np.array([[-0.5, 0.5],
                                                      [0.5, 0.5]])),
          '(cf. exact -/+ 1/8)')

    ax = mesh.plot(pressure)
    ax.axis('off')
    ax.get_figure().savefig(f'{name}_pressure.png')

    ax = mesh.draw()
    velocity1 = velocity[basis['u'].nodal_dofs]
    ax.quiver(mesh.p[0, :], mesh.p[1, :],
              velocity1[0, :], velocity1[1, :],
              mesh.p[0, :])         # colour by buoyancy
    ax.axis('off')
    ax.get_figure().savefig(f'{name}_velocity.png')




    ax = mesh.draw()
    ax.tricontour(Triangulation(mesh.p[0, :], mesh.p[1, :], mesh.t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])
    ax.axis('off')
    ax.get_figure().savefig(f'{name}_stream-function.png')
