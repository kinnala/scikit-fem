from skfem import *
from skfem.models.poisson import vector_laplace, laplace, mass
from skfem.models.general import divergence, rot

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, minres

mesh = MeshQuad.init_tensor(*(np.linspace(-.5, .5, 2**6),)*2)

element = {'u': ElementVectorH1(ElementQuad2()),
           'p': ElementQuad1()}
basis = {variable: InteriorBasis(mesh, e, intorder=3)
         for variable, e in element.items()}


@LinearForm
def body_force(v, w):
    return w.x[0] * v.value[1]


A = asm(vector_laplace, basis['u'])
B = -asm(divergence, basis['u'], basis['p'])
f = asm(body_force, basis['u'])
D = basis['u'].find_dofs()['all'].all()
Aint = condense(A, D=D, expand=False)
solver = solver_iter_pcg(M=build_pc_ilu(Aint))
I = basis['u'].complement_dofs(D)


def flow(pressure: np.ndarray) -> np.ndarray:
    """compute the velocity corresponding to a guessed pressure"""
    velocity = np.zeros(basis['u'].N)
    velocity[I] = solve(Aint,
                        condense(csr_matrix(A.shape),
                                 f - B.T @ pressure, I=I)[1],
                        solver=solver)
    return velocity


def dilatation(pressure: np.ndarray) -> np.ndarray:
    """compute the dilatation corresponding to a guessed pressure"""
    return -B @ flow(pressure)


pressure = np.zeros(basis['p'].N)
dilatation0 = dilatation(pressure)

K = LinearOperator((basis['p'].N,) * 2,
                   lambda p: dilatation(p) - dilatation0,
                   dtype=pressure.dtype)

pressure = solve(K, -dilatation0,
                 solver=solver_iter_krylov(minres),
                 M=build_pc_diag(asm(mass, basis['p'])))

velocity = flow(pressure)

basis['psi'] = InteriorBasis(mesh, ElementQuad2())
psi = np.zeros(A.shape[0])
vorticity = asm(rot, basis['psi'],
                w=[basis['psi'].interpolate(velocity[i::2])
                   for i in range(2)])
psi = solve(*condense(asm(laplace, basis['psi']), vorticity, D=basis['psi'].find_dofs()))
psi0 = basis['psi'].interpolator(psi)(np.zeros((2, 1)))[0]


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    from matplotlib.tri import Triangulation
    from skfem.visuals.matplotlib import plot, draw

    print(psi0)

    name = splitext(argv[0])[0]
    plot(mesh, pressure, colorbar=True).get_figure().savefig(
        f'{name}_pressure.png')

    ax = draw(mesh)
    ax.tricontour(Triangulation(*mesh.p, mesh._splitquads().t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])
    ax.get_figure().savefig(f'{name}_stream-lines.png')
