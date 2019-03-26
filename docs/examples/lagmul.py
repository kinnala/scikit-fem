from skfem import *
from skfem.models.poisson import laplace, mass

from scipy.sparse import bmat

import numpy as np
mesh = MeshTri()
mesh.refine(4)

element = ElementTriP1()
basis = {'interior': InteriorBasis(mesh, element),
         'facet': FacetBasis(mesh, element, facets=mesh.boundary_facets())}

A = asm(laplace, basis['interior'])
B = asm(mass, basis['facet'])
K = bmat([[A, B], [B.T, None]]).tocsr()


@linear_form
def dirichlet(v, dv, w):
    return v * ((w.x[0] + 1.j * w.x[1])**2).real


f = np.concatenate([np.zeros(basis['interior'].N),
                    asm(dirichlet, basis['facet'])])

u = solve(K, f, solver_iter_pcg())
print(u.shape)
u = u[basis['interior'].N:]
print(u.shape)


