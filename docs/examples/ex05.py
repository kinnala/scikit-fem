from pathlib import Path
from skfem import *

m = MeshTri()
m.refine(5)

e = ElementTriP1()

ib = InteriorBasis(m, e)
fb = FacetBasis(m, e)

@bilinear_form
def bilinf(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@bilinear_form
def facetbilinf(u, du, v, dv, w):
    n = w.n
    x = w.x
    return -(du[0]*n[0] + du[1]*n[1])*v*(x[0] == 1.0)

@linear_form
def facetlinf(v, dv, w):
    n = w.n
    x = w.x
    return -(dv[0]*n[0] + dv[1]*n[1])*(x[0] == 1.0)

A = asm(bilinf, ib)
B = asm(facetbilinf, fb)

b = asm(facetlinf, fb)

D = ib.get_dofs(lambda x: (x[1] == 0.0)).all()
I = ib.complement_dofs(D)

import scipy.sparse
b = scipy.sparse.csr_matrix(b)
K = scipy.sparse.bmat([[A+B, b.T], [b, None]]).tocsr()

import numpy as np
f = np.concatenate((np.zeros(A.shape[0]), -1.0*np.ones(1)))

I = np.append(I, K.shape[0] - 1)

x = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    
    m.plot(x[:-1], colorbar=True)
    m.savefig(splitext(argv[0])[0] + '_solution.png')
