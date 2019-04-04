from skfem import *
from skfem.models.poisson import laplace, mass

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


u_exact = L2_projection(dirichlet, basis)
u = u_exact.copy()
u[I] = solve(*condense(A, np.zeros_like(u), u, I))


if __name__ == "__main__":

    from os.path import splitext
    from sys import argv

    u_error = u - u_exact
    print('L2 error = ', np.sqrt(u_error.T @ (asm(mass, basis) @ u_error)))

    m.plot(u[basis.nodal_dofs.flatten()])
    m.savefig(splitext(argv[0])[0] + '_solution.png')
