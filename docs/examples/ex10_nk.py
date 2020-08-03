"""Nonlinear Poisson equation.

This example solves the nonlinear minimal surface problem using a preconditioned Jacobian-free Newton-Krylov method.

The idea is (Knoll & Keyes 2004, &sect;5.1)

    JFNK performs the Newton–Krylov iteration matrix-free, while the Jacobian used in the preconditioner is formed with a frequency of p Newton iterations.

This is to be contrasted with (ibid.)

    MNK (modified Newton–Krylov) is a standard Newton–Krylov method but the Jacobian is formed with a frequency of p Newton iterations for use in both the matrix vector multiply and the preconditioner.

The advantage of the present method is that the finite-difference Fréchet approximation of the Jacobian in the Newton iteration is always accuate and up to date, only the quality of the preconditioner degrades with lagging.

The Jacobian-free Newton-Krylov iteration is implemented in `scipy.optimize.root(method="krylov")` and an incomplete LU factorization of the lagged Jacobian is used as preconditioner.  The Jacobian is only reassembled and refactorized every p Newton iterations, and here by default p is infinite, in which case only the Jacobian based on the initial guess is ever used.

"""

from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import laplace
import numpy as np
from scipy.optimize import root
from scipy.sparse.linalg import LinearOperator, spilu
from typing import List, Optional

m = MeshTri()
m.refine(5)


@BilinearForm
def jacobian(u, v, w):
    return (1 / np.sqrt(1 + dot(grad(w['w']), grad(w['w']))) * dot(grad(u), grad(v))
            - 2 * dot(grad(u), grad(w['w'])) * dot(grad(w['w']), grad(v))
            / 2 / (1 + dot(grad(w['w']), grad(w['w'])))**(3/2))


@LinearForm
def rhs(v, w):
    return dot(grad(w['w']), grad(v)) / np.sqrt(1 + dot(grad(w['w']), grad(w['w'])))


basis = InteriorBasis(m, ElementTriP1())

x = np.zeros(basis.N)

I = m.interior_nodes()
D = m.boundary_nodes()
x[D] = np.sin(np.pi * m.p[0, D])
M = LinearOperator([len(I)]*2, matvec=lambda u: u)


def residual(u: np.ndarray) -> np.ndarray:
    x[I] = u
    return asm(rhs, basis, w=basis.interpolate(x))[I]

    
def update(u: np.ndarray,
           _: Optional[np.ndarray] = None,
           updates: List[int] = [0],
           period: int = np.inf) -> None:

    if updates[0] % period == 0:
        print('Updating Jacobian preconditioner.')
        x[I] = u
        J = asm(jacobian, basis, w=basis.interpolate(x))
        JI = condense(J, D=D, expand=False).tocsc()
        M.matvec = spilu(JI).solve

    updates[0] += 1


M.update = update
M.update(x[I])

sol = root(residual, x[I], method='krylov',
           options={'disp': True,
                    'jac_options': {
                        'inner_M': M}})

x[I] = sol.x


if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3, show
    plot3(m, x)
    show()
