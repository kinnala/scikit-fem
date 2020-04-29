from skfem import *
from skfem.models.poisson import laplace

import numpy as np

m = MeshTri()
m.refine(4)

e = ElementTriP2()
basis = InteriorBasis(m, e)

A = asm(laplace, basis)


def dirichlet(x, y):
    """return a harmonic function"""
    return ((x + 1.j * y) ** 2).real


boundary_basis = FacetBasis(m, e)
boundary_dofs = boundary_basis.find_dofs()['all'].all()

u = np.zeros(basis.N)
u[boundary_dofs] = L2_projection(dirichlet, boundary_basis, boundary_dofs)
u = solve(*condense(A, np.zeros_like(u), u, D=boundary_dofs))


if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    print('||grad u||**2 = {:f} (exact = 8/3 = {:f})'.format(u @ A @ u, 8/3))
    plot(basis, u)
    show()
