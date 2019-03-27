from skfem import *
from skfem.models.poisson import laplace, mass

from scipy.sparse import bmat

import numpy as np
mesh = MeshTri()
mesh.refine(4)

element = {'1': ElementTriP1(), '2': ElementTriP2()}
basis = {'interior': InteriorBasis(mesh, element['2']),
         'facet': FacetBasis(mesh, element['1'])}

A = asm(laplace, basis['interior'])
B = asm(mass,
        FacetBasis(mesh, ElementTriP2(), intorder=3),
        FacetBasis(mesh, ElementTriP1(), intorder=3))[mesh.boundary_nodes()]
K = bmat([[A, B.T], [B, None]]).tocsr()


def exact(x, y):
    return ((x + 1.j * y)**2).real


@linear_form
def dirichlet(v, dv, w):
    return v * exact(*w.x)


f = np.concatenate(
    [np.zeros(basis['interior'].N),
     asm(dirichlet,
         FacetBasis(mesh, ElementTriP1(), intorder=3))[mesh.boundary_nodes()]])

u, p = np.split(solve(K, f), [basis['interior'].N])
u_exact = L2_projection(exact, basis['interior'])


if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    u_error = u - u_exact
    print('L2 error = ',
          np.sqrt(u_error.T @ (asm(mass, basis['interior']) @ u_error)))
    
    mesh.plot(u[basis['interior'].nodal_dofs.flatten()])
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
