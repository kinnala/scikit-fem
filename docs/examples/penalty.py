"""

Solve the Dirichlet problem again, this time with a penalty method (e.g. Carey & Oden 1983, §3.5.2; Ern & Guermond 2004, pp. 379–380; Juntunen & Stenberg 2009).

* Carey, G. F., & Oden, J. T. (1983). *Finite Elements: A Second Course.*  Englewood Cliffs, New Jersey: Prentice-Hall

* Ern, A., & Guermond, J.-L. (2004). *Theory and Practice of Finite Elements*. New York: Springer

* Juntunen, M. & Stenberg, R. (2009). Nitsche's method for general boundary conditions. *Mathematics of Computation,* **78**:1353–1374, eq. 2.12

"""

from skfem import *
import numpy as np

from lagmul import mass, basis, A, mesh, dirichlet_forcing, u_exact

basis['boundary'] = FacetBasis(mesh, ElementTriP2(),
                               facets=mesh.boundary_facets())
epsilon = 1e-6
penalty = asm(mass, basis['boundary'])

b = asm(dirichlet_forcing, basis['boundary'])
u = solve(A + penalty / epsilon, b / epsilon)


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    u_error = u - u_exact
    print('L2 error = ',
          np.sqrt(u_error.T @ (asm(mass, basis['interior']) @ u_error)))

    mesh.plot(u[basis['interior'].nodal_dofs.flatten()])
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
