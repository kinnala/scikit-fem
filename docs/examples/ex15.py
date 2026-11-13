"""One-dimensional Poisson."""

import numpy as np
from skfem import *
from skfem.models.poisson import laplace, unit_load

m = MeshLine(np.linspace(0, 1, 10))

e = ElementLineP1()
basis = Basis(m, e)

A = asm(laplace, basis)
b = asm(unit_load, basis)

x = solve(*condense(A, b, D=basis.get_dofs()))

def visualize():
    from skfem.visuals.web import plot
    return plot(basis, x)

if __name__ == "__main__":
    visualize().show()
