r"""Krylov–Uzawa method for the Stokes equation.

This example solves the Stokes equation iteratively in a square domain.

Direct solvers for the Stokes equation do not scale; iterative solvers are
required for all but the smallest two-dimensional problems.  A square domain is
used here instead of the disk since it easier to generate the finer mesh
required to demonstrate the time taken for a slightly larger problem.

A classical iterative procedure for the Stokes equation is the Uzawa conjugate
gradient method in which an auxiliary operator is formed which maps a guessed
pressure field :math:`p` to the divergence

.. math::
   \Theta (p) = \nabla\cdot \{\boldsymbol{u} (p)\}
   
of the velocity field which satisfies the momentum equation with that pressure
[GLOWINSKI-PIRONNEAU]_ [BRAESS]_.

.. math::
    \nu\Delta\boldsymbol{u} = -\rho^{-1}\nabla p + \boldsymbol{f}

This operator :math:`\Theta(p)` is affine but subtracting the divergence given
by zero pressure gives a linear operator

.. math::
   K (p) = \Theta (p) - \Theta (0)
   
for which the linear operator equation

.. math::
   K (p) = -\Theta (0)
   
can be solved by a Krylov method, classically the method of conjugate gradients.
 The solution is the pressure :math:`p` that gives the velocity
:math:`\boldsymbol u` that has zero divergence.

At each iteration, the above vector Poisson equation is solved for a velocity
field.  As the problem gets larger, this too should be solved iteratively.  The
method of conjugate gradients could be used for this too, but a catch is that
:func:`scipy.sparse.linalg.cg` is not re-entrant and so cannot be used at both
levels.  This is simply avoided here by using
:func:`scipy.sparse.linalg.minres` for the pressure.

The results may be assessed using the value of the stream-function at the
centre.  As in :ref:`biharmonic`, the stream-function satisfies the same
boundary value problem as the deflection of a clamped plate, for which the
central deflection is known to be approximately 0.162/128 [LOVE]_.

.. [BRAESS] Braess, D. (2001). *Finite Elements.* Cambridge University Press, §IV.5.1
  
.. [GLOWINSKI-PIRONNEAU] Glowinski, R. & Pironneau, O. (1979). On numerical methods for the Stokes problem. In R. Glowinski, E. Y. Rodin & O. C. Zienkiewicz (eds.), *Energy Methods in Finite Element Analysis* (pp. 243–264), Wiley
.. [LOVE] Love, A. E. H. (1944). *A Treatise on the Mathematical Theory of Elasticity.* Dover

"""
from skfem import *
from skfem.models.poisson import vector_laplace, laplace, mass
from skfem.models.general import divergence, rot

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, minres

mesh = MeshQuad.init_tensor(*(np.linspace(-.5, .5, 2**6),)*2)

element = {'u': ElementVectorH1(ElementQuad2()),
           'p': ElementQuad1()}
basis = {variable: Basis(mesh, e, intorder=3)
         for variable, e in element.items()}


@LinearForm
def body_force(v, w):
    return w.x[0] * v.value[1]


A = asm(vector_laplace, basis['u'])
B = -asm(divergence, basis['u'], basis['p'])
f = asm(body_force, basis['u'])
D = basis['u'].get_dofs()
Aint = condense(A, D=D, expand=False)
solver = solver_iter_pcg(M=build_pc_ilu(Aint))
I = basis['u'].complement_dofs(D)


def flow(pressure: np.ndarray) -> np.ndarray:
    """compute the velocity corresponding to a guessed pressure"""
    velocity = basis['u'].zeros()
    velocity[I] = solve(Aint,
                        condense(csr_matrix(A.shape),
                                 f - B.T @ pressure, I=I)[1],
                        solver=solver)
    return velocity


def dilatation(pressure: np.ndarray) -> np.ndarray:
    """compute the dilatation corresponding to a guessed pressure"""
    return -B @ flow(pressure)


dilatation0 = dilatation(basis['p'].zeros())

K = LinearOperator((basis['p'].N,) * 2,
                   lambda p: dilatation(p) - dilatation0,
                   dtype=dilatation0.dtype)

pressure = solve(K, -dilatation0,
                 solver=solver_iter_krylov(minres),
                 M=build_pc_diag(asm(mass, basis['p'])))

velocity = flow(pressure)

basis['psi'] = basis['u'].with_element(ElementQuad2())
vorticity = asm(rot, basis['psi'], w=basis['u'].interpolate(velocity))
psi = solve(*condense(asm(laplace, basis['psi']), vorticity, D=basis['psi'].get_dofs()))
psi0 = (basis['psi'].probes(np.zeros((2, 1))) @ psi)[0]


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
    ax.tricontour(Triangulation(*mesh.p, mesh.to_meshtri().t.T),
                  psi[basis['psi'].nodal_dofs.flatten()])
    ax.get_figure().savefig(f'{name}_stream-lines.png')
