"""
Author: kinnala

Impose integral condition using a Lagrange multiplier.
"""
from skfem import *

m = MeshTri()
m.refine(5)

e = ElementTriP1()
map = MappingAffine(m)

ib = InteriorBasis(m, e, map, 2)
fb = FacetBasis(m, e, map, 2)

@bilinear_form
def bilinf(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@bilinear_form
def facetbilinf(u, du, v, dv, w):
    n = w[2]
    x = w[0]
    return -(du[0]*n[0] + du[1]*n[1])*v*(x[0] == 1.0)

@linear_form
def facetlinf(v, dv, w):
    n = w[2]
    x = w[0]
    return -(dv[0]*n[0] + dv[1]*n[1])*(x[0] == 1.0)

A = asm(bilinf, ib)
B = asm(facetbilinf, fb)

b = asm(facetlinf, fb)

_, D = ib.find_dofs(lambda x, y: (y == 0.0))
I = ib.dofnum.complement_dofs(D)

import scipy.sparse
b = scipy.sparse.csr_matrix(b)
K = scipy.sparse.bmat([[A+B, b.T], [b, None]]).tocsr()

import numpy as np
f = np.concatenate((np.zeros(A.shape[0]), -1.0*np.ones(1)))

x = np.zeros(K.shape[0])

I = np.append(I, K.shape[0]-1)

x[I] = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    m.plot3(x[:-1])
    m.show()
