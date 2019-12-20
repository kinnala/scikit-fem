from skfem import *
from skfem.models.poisson import vector_laplace, laplace
from skfem.models.general import divergence, rot

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import cg, LinearOperator, minres

mesh = MeshQuad.init_tensor(*(np.linspace(-.5, .5, 2**6),)*2)

element = {'u': ElementVectorH1(ElementQuad2()),
           'p': ElementQuad1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@linear_form
def body_force(v, dv, w):
    return w.x[0] * v[1]


A = asm(vector_laplace, basis['u'])
B = asm(divergence, basis['u'], basis['p'])
f = asm(body_force, basis['u'])
D = basis['u'].get_dofs().all()
Aint = condense(A, D=D, expand=False)
solver = solver_iter_krylov(cg, M=build_pc_ilu(Aint))
I = basis['u'].complement_dofs(D)



def flow(pressure: np.ndarray) -> np.ndarray:
    """compute the velocity corresponding to a guessed pressure"""
    velocity = np.zeros(basis['u'].N)
    velocity[I] = solve(Aint,
                        condense(csr_matrix(A.shape),
                                 f + B.T @ pressure, I=I)[1],
                        solver=solver)
    return velocity


def dilatation(pressure: np.ndarray) -> np.ndarray:
    """compute the dilatation corresponding to a guessed pressure"""
    return B @ flow(pressure)


pressure = np.zeros(basis['p'].N)
dilatation0 = dilatation(pressure)

K = LinearOperator((basis['p'].N,) * 2,
                   lambda p: dilatation(p) - dilatation0,
                   dtype=pressure.dtype)

pressure, info = minres(K, -dilatation0)

if info != 0:
    raise RuntimeError('conjugate gradient '
                       f'not converging after {info} iterations'
                       if info > 0 else
                       'illegal input or breakdown')

velocity = flow(pressure)

basis['psi'] = InteriorBasis(mesh, ElementQuad2())
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

    mesh.save(f'{name}_velocity.vtk', {'u': velocity[basis['u'].nodal_dofs].T})

    print(basis['psi'].interpolator(psi)(np.zeros((2, 1)))[0])

    print(basis['p'].interpolator(pressure)(np.array([[-0.5, 0.5],
                                                      [0.5, 0.5]])))

    mesh.plot(pressure, colorbar=True).get_figure().savefig(
        f'{name}_pressure.png')

    ax = mesh.draw()
    velocity1 = velocity[basis['u'].nodal_dofs]
    ax.quiver(mesh.p[0, :], mesh.p[1, :],
              velocity1[0, :], velocity1[1, :],
              mesh.p[0, :])         # colour by buoyancy
    ax.get_figure().savefig(f'{name}_velocity.png')

    ax = mesh.draw()
    ax.tricontour(Triangulation(*mesh.p, mesh._splitquads().t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])
    ax.get_figure().savefig(f'{name}_stream-function.png')
