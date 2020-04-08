from skfem import *
from skfem.helpers import dot, grad
from skfem.models.poisson import laplace

m = MeshTri()
m.refine(5)

e = ElementTriP1()

ib = InteriorBasis(m, e)
fb = FacetBasis(m, e)


@BilinearForm
def facetbilinf(u, v, w):
    n = w.n
    x = w.x
    return -dot(grad(u), n) * v * (x[0] == 1.0)

@LinearForm
def facetlinf(v, w):
    n = w.n
    x = w.x
    return -dot(grad(v), n) * (x[0] == 1.0)

A = asm(laplace, ib)
B = asm(facetbilinf, fb)

b = asm(facetlinf, fb)

D = ib.get_dofs(lambda x: (x[1] == 0.0)).all()
I = ib.complement_dofs(D)

import scipy.sparse
b = scipy.sparse.csr_matrix(b)
K = scipy.sparse.bmat([[A+B, b.T], [b, None]], 'csr')

import numpy as np
f = np.concatenate((np.zeros(A.shape[0]), -1.0*np.ones(1)))

I = np.append(I, K.shape[0] - 1)

x = solve(*condense(K, f, I=I))

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import plot, savefig
    plot(m, x[:-1], colorbar=True)
    savefig(splitext(argv[0])[0] + '_solution.png')
