"""Interior penalty method."""

from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import laplace, unit_load

m = MeshTri.init_sqsymmetric().refined()
mapping = MappingAffine(m)
e = ElementTriDG(ElementTriP1())
alpha = 1e-1

ib = Basis(m, e)
fb = {}
fb[0] = InteriorFacetBasis(m, e, side=0)
fb[1] = InteriorFacetBasis(m, e, side=1)
bb = FacetBasis(m, e)

@BilinearForm
def bilin_bnd(u, v, w):
    h = w.h
    n = w.n
    return (u * v) / alpha / h - dot(grad(u), n) * v - u * dot(grad(v), n)

A = asm(laplace, ib)
b = asm(unit_load, ib)

C = asm(bilin_bnd, bb)
B = 0
for i in range(2):
    for j in range(2):
        @BilinearForm
        def bilin_int(u, v, w):
            ju = (-1.0)**i*u
            jv = (-1.0)**j*v
            n = w.n
            h = w.h
            return (ju * jv) / alpha / h - (dot(grad(u), n) * jv +
                                            dot(grad(v), n) * ju) / 2

        B = asm(bilin_int, fb[i], fb[j]) + B

x = solve(A+B+C, b)

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    plot(M, X, shading="gouraud", colorbar=True)
    show()
