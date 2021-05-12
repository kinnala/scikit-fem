r"""Minimal surface

This example reformulates the minimal surface problem of ex10 as a minimization
and uses scipy.optimize.minimize.

"""

from .ex10 import jacobian, rhs, basis, D
from skfem import *
from skfem.helpers import dot, trace
import numpy as np
from scipy.optimize import LinearConstraint, minimize
from scipy.sparse import csr_matrix


@Functional
def area(w):
    return np.sqrt(1 + dot(w["w"].grad, w["w"].grad))


def fun(z: np.ndarray) -> float:
    return area.assemble(basis, w=basis.interpolate(z))


def jac(z: np.ndarray) -> np.ndarray:
    return rhs.assemble(basis, w=basis.interpolate(z))


def hess(z: np.ndarray) -> np.ndarray:
    return jacobian.assemble(basis, w=basis.interpolate(z))


trace = csr_matrix((np.ones(D.size), (np.arange(D.size), D)), (D.size, basis.N))
constraint = LinearConstraint(trace, *[np.sin(np.pi * basis.mesh.p[0, D])]*2)

x = minimize(
    fun,
    basis.zeros(),
    jac=jac,
    hess=hess,
    method="trust-constr",
    constraints=constraint,
).x

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot3, show
    plot3(basis, x)
    show()
