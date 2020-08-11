r"""Bratu–Gelfand.

.. note::

   This example requires the external package `pacopy 0.1.2 <https://pypi.org/project/pacopy/0.1.2/>`_

Here the bifurcation diagram for the Bratu–Gelfand two-point boundary value problem is reproduced by numerical continuation as implemented in pacopy, and adapted from the pacopy example `Bratu <https://github.com/nschloe/pacopy/blob/v0.1.2/README.md#bratu>`_.

.. math::
    u'' + \lambda \mathrm e^u = 0, \quad 0 < x < 1,
with :math:`u(0)=u(1)=0` and where :math:`\lambda > 0` is a parameter.

For treatment by numerical continuation, we define the residual

.. math::
    F(u, \lambda) = -u'' - \lambda \mathrm e^u
.. literalinclude:: ex23.py
    :start-at: def f
    :lines: 1-5

its derivative with respect to the parameter

.. math::
   \frac{\partial F}{\partial\lambda} = -\mathrm e^u
.. literalinclude:: ex23.py
    :start-at: def df_dlmbda
    :lines: 1-8

and the Jacobian

.. math::
   J (u) = -\frac{\mathrm d^2}{\mathrm dx^2} - \lambda \mathrm e^u
.. literalinclude:: ex23.py
    :start-at: def jacobian_solver
    :lines: 1-11

The resulting bifurcation diagram, matches figure 1.1 (left) of Farrell, Birkisson, & Funke (2015).

.. figure:: ex23.png


* P. E. Farrell, Á. Birkisson, & S. W. Funke (2015). Deflation techniques for finding distinct solutions of nonlinear partial differential equations. *SIAM Journal on Scientific Computing* 37(4). pp. A2026–A2045. `doi:10.1137/140984798 <http://dx.doi.org/10.1137/140984798>`_

"""
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
        self.D = self.basis.find_dofs()['all'].nodal['u']

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

turning_point = 3.51383         # Farrell et al

try:
    euler_newton(
        problem, u0, lmbda0, callback, max_steps=500, newton_tol=1.0e-10
    )
except RangeException:

    if __name__ == '__main__':
        fig, ax = subplots()
        ax.set_xlabel(r'$\lambda$')
        ax.set_ylabel(r'$||u||_2$')
        ax.grid()
        ax.plot(lmbda_list, values_list, '-o')
        ax.axvline(turning_point, linestyle='dotted')
        ax.set_xlim(0.0, 4.0)
        ax.set_ylim(0.0, upper)
        fig.savefig(Path(__file__).with_suffix('.png'))
