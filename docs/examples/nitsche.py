"""

Solve the Dirichlet problem again, this time with Nitsche's method (Juntunen & Stenberg 2009).

* Juntunen, M. & Stenberg, R. (2009). Nitsche's method for general boundary conditions. *Mathematics of Computation,* **78**:1353–1374, eq. 2.12

"""

from skfem import *
from skfem.models.poisson import mass

import numpy as np

from ex14 import basis, A, u_exact, dirichlet


fbasis = FacetBasis(basis.mesh, basis.elem)

alpha = 1e-3                    # TODO: How is alpha chosen?


@bilinear_form
def nitsche_bilinear(u, du, v, dv, w):
    return u * v / w.h / alpha - u * sum(w.n * dv) - sum(w.n * du) * v


@linear_form
def nitsche_linear(v, dv, w):
    u0 = dirichlet(*w.x)
    return u0 * v / w.h / alpha - u0 * sum(w.n * dv)

    
B = A + asm(nitsche_bilinear, fbasis)
L = asm(nitsche_linear, fbasis)

u = solve(B, L)


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    u_error = u - u_exact
    print('L2 error = ',
          np.sqrt(u_error.T @ (asm(mass, basis) @ u_error)))

    basis.mesh.plot(u[basis.nodal_dofs.flatten()])
    basis.mesh.savefig(splitext(argv[0])[0] + '_solution.png')
