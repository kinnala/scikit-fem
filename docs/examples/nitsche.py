"""

Solve the Dirichlet problem again, this time with Nitsche's method (Juntunen & Stenberg 2009).

* Juntunen, M. & Stenberg, R. (2009). Nitsche's method for general boundary conditions. *Mathematics of Computation,* **78**:1353–1374, eq. 2.12

"""

from skfem import *
from skfem.models.poisson import mass

import numpy as np

from ex14 import basis, A, u_exact

fbasis = FacetBasis(basis.mesh, basis.elem,
                    facets=basis.mesh.boundary_facets())

alpha = 1e-1                      # TODO: How is alpha chosen?


@bilinear_form
def nitsche_bilinear(u, du, v, dv, w):
    return (u * v) / w.h / alpha - sum(w.n * du) * v - sum(w.n * dv) * u


@linear_form
def nitsche_linear(v, dv, w):
    u0 = dirichlet(*w.x)
    return u0 * v / w.h / alpha - sum(w.n * dv) * u0

    
B = A + asm(nitsche_bilinear, fbasis)
L = asm(nitsche_linear, fbasis)

u = solve(B, L)


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    u_error = u - u_exact
    print('L2 error = ',
          np.sqrt(u_error.T @ (asm(mass, basis) @ u_error)))

    mesh.plot(u[basis.nodal_dofs.flatten()])
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
