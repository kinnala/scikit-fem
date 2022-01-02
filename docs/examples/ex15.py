"""# One-dimensional Poisson equation

"""
import numpy as np
from skfem import *
from skfem.models.poisson import laplace, unit_load

m = MeshLine(np.linspace(0, 1, 10))

e = ElementLineP1()
basis = Basis(m, e)

A = asm(laplace, basis)
b = asm(unit_load, basis)

x = solve(*condense(A, b, D=basis.get_dofs()))

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    plot(m, x)
    show()
