from skfem import *
from skfem.models.poisson import vector_laplace, laplace
from skfem.models.general import divergence, rot


from itertools import count

import numpy as np
from scipy.sparse import csr_matrix

from sksparse.cholmod import cholesky


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


alpha = 2e2                     # constant Uzawa step-size
pressure = np.zeros(basis['p'].N)

velocity = np.zeros(basis['u'].N)
A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
f = asm(body_force, basis['u'])
D = basis['u'].get_dofs().all()
backsolve = cholesky(condense(A, D=D, expand=False).T)  # cholesky prefers CSC
I = basis['u'].complement_dofs(D)


def flow(pressure: np.ndarray) -> np.ndarray:
    """compute the velocity corresponding to a guessed pressure"""
    velocity[I] = backsolve(condense(csr_matrix(A.shape),
                                     f + B.T @ pressure, I=I)[1])
    return velocity


def dilatation(pressure: np.ndarray) -> np.ndarray:
    """compute the dilatation corresponding to a guessed pressure"""
    return B @ flow(pressure)


for iteration in count():

    theta = dilatation(pressure)
    pressure -= alpha * theta

    dilatation_norm = np.linalg.norm(theta)
    print(iteration, dilatation_norm)
    if dilatation_norm < 1e-4:
        break


basis['psi'] = InteriorBasis(mesh, ElementTriP2())
psi = np.zeros(A.shape[0])
D = basis['psi'].get_dofs().all()
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi = solve(*condense(asm(laplace, basis['psi']), vorticity, D=D))


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

    mesh.plot(pressure, colorbar=True).get_figure().savefig(
        f'{name}_pressure.png')

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
