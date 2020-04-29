from skfem import *
from skfem.models.poisson import vector_laplace, mass, laplace
from skfem.models.general import divergence, rot

import numpy as np
from scipy.sparse import bmat

import dmsh

mesh = MeshTri(*map(np.transpose,
                    dmsh.generate(dmsh.Circle([0., 0.], 1.), .1)))

element = {'u': ElementVectorH1(ElementTriP2()),
           'p': ElementTriP1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@LinearForm
def body_force(v, w):
    return w.x[0] * v.value[1]


A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
C = asm(mass, basis['p'])

K = bmat([[A, -B.T],
          [-B, 1e-6 * C]], 'csr')

f = np.concatenate([asm(body_force, basis['u']),
                    np.zeros(B.shape[0])])

uvp = solve(*condense(K, f, D=basis['u'].find_dofs()))

velocity, pressure = np.split(uvp, [A.shape[0]])

basis['psi'] = InteriorBasis(mesh, ElementTriP2())
A = asm(laplace, basis['psi'])
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi = solve(*condense(A, vorticity, D=basis['psi'].find_dofs()))


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation

    from skfem.visuals.matplotlib import plot, draw, savefig

    name = splitext(argv[0])[0]

    mesh.save(f'{name}_velocity.vtk',
              {'velocity': velocity[basis['u'].nodal_dofs].T})

    print(basis['psi'].interpolator(psi)(np.zeros((2, 1)))[0],
          '(cf. exact 1/64)')

    print(basis['p'].interpolator(pressure)(np.array([[-0.5, 0.5],
                                                      [0.5, 0.5]])),
          '(cf. exact -/+ 1/8)')

    ax = draw(mesh)
    plot(basis['p'], pressure, ax=ax)
    savefig(f'{name}_pressure.png')

    ax = draw(mesh)
    velocity1 = velocity[basis['u'].nodal_dofs]
    ax.quiver(*mesh.p, *velocity1, mesh.p[0, :])  # colour by buoyancy
    savefig(f'{name}_velocity.png')

    ax = draw(mesh)
    ax.tricontour(Triangulation(*mesh.p, mesh.t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])
    savefig(f'{name}_stream-function.png')
