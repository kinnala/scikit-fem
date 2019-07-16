from skfem import *
from skfem.models.poisson import laplace

import numpy as np

m = MeshTri()
m.refine(4)

e = ElementTriP2()
basis = InteriorBasis(m, e)

A = asm(laplace, basis)

D = basis.get_dofs().all()
I = basis.complement_dofs(D)


def dirichlet(x, y):
    """return a harmonic function"""
    return ((x + 1.j * y) ** 2).real


u = np.zeros(basis.N)
u[D] = dirichlet(*basis.doflocs[:, D])
u[I] = solve(*condense(A, np.zeros_like(u), u, I))


if __name__ == "__main__":
    print('||grad u||**2 = {:f} (exact = 8/3 = {:f})'.format(u @ A @ u, 8/3))
    m.plot(u[basis.nodal_dofs.flatten()])
    m.show()
