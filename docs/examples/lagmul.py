from skfem import *
from skfem.models.poisson import mass

from scipy.sparse import bmat

import numpy as np

from ex14 import basis as interior_basis
from ex14 import dirichlet, A


mesh = interior_basis.mesh
basis = {'interior': interior_basis,
         'interior-facet': FacetBasis(mesh, ElementTriP2(), intorder=3),
         'facet': FacetBasis(mesh, ElementTriP1(), intorder=3)}

B = asm(mass, basis['interior-facet'], basis['facet'])[mesh.boundary_nodes()]
K = bmat([[A, B.T], [B, None]]).tocsr()


@linear_form
def dirichlet_forcing(v, dv, w):
    return v * dirichlet(*w.x)


f = np.concatenate(
    [np.zeros(basis['interior'].N),
     asm(dirichlet_forcing, basis['facet'])[mesh.boundary_nodes()]])

u, p = np.split(solve(K, f), [basis['interior'].N])
u_exact = L2_projection(dirichlet, basis['interior'])


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    u_error = u - u_exact
    print('L2 error = ',
          np.sqrt(u_error.T @ (asm(mass, basis['interior']) @ u_error)))

    mesh.plot(u[basis['interior'].nodal_dofs.flatten()])
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
