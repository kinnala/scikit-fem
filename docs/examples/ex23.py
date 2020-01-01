from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse import dia_matrix

from pacopy import euler_newton
from skfem import *
from skfem.models.poisson import laplace, mass


class Bratu1d:

    def __init__(self, n: int):

        self.basis = InteriorBasis(MeshLine(np.linspace(0, 1, n)),
                                   ElementLineP1())
        self.lap = asm(laplace, self.basis)
        self.mass = asm(mass, self.basis)
        self.D = self.basis.get_dofs().all()

    def inner(self, a: np.ndarray, b: np.ndarray) -> float:
        """return the inner product of two solutions"""
        return a.T @ (self.mass @ b)

    def norm2_r(self, a: np.ndarray) -> float:
        """return the squared norm in the range space

        used to determine if a solution has been found.
        """
        return a.T @ a

    def f(self, u: np.ndarray, lmbda: float) -> np.ndarray:
        """return the residual at u"""
        out = self.lap @ u - lmbda * self.mass @ np.exp(u)
        out[self.D] = u[self.D]
        return out

    def df_dlmbda(self, u: np.ndarray, lmbda: float) -> np.ndarray:
        """The derivative of the residual with respect to the parameter.

        Used in Euler-Newton continuation.
        """
        out = -self.mass @ np.exp(u)
        out[self.D] = 0.0
        return out

    def jacobian_solver(self,
                        u: np.ndarray,
                        lmbda: float,
                        rhs: np.ndarray) -> np.ndarray:
        """Solver for the Jacobian problem."""
        A = self.lap - lmbda * dia_matrix((self.mass @ np.exp(u), 0),
                                          self.mass.shape)
        du = np.zeros_like(u)
        du = solve(*condense(A, rhs, D=self.D))
        return du


problem = Bratu1d(2**8)
u0 = np.zeros(problem.basis.N)
lmbda0 = 0.0

lmbda_list = []
values_list = []

upper = 6.


class RangeException(Exception):
    pass


def callback(k, lmbda, sol):

    lmbda_list.append(lmbda)
    values_list.append(np.sqrt(problem.inner(sol, sol)))
    if values_list[-1] > upper:
        raise RangeException


try:
    euler_newton(
        problem, u0, lmbda0, callback, max_steps=500, newton_tol=1.0e-10
    )
except RangeException:
    fig, ax = subplots()
    ax.set_xlabel('$\lambda$')
    ax.set_ylabel('$||u||_2$')
    ax.grid()
    ax.plot(lmbda_list, values_list, '-o')
    ax.axvline(3.51383, linestyle='dotted')  # turning point (Farrell et al)
    ax.set_xlim(0.0, 4.0)
    ax.set_ylim(0.0, upper)
    fig.savefig(Path(__file__).with_suffix('.png'))
