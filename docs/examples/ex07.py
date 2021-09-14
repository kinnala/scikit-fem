"""Interior penalty method."""

from skfem import *
from skfem.helpers import grad, dot
from skfem.models.poisson import laplace, unit_load

m = MeshTri.init_sqsymmetric().refined()
e = ElementTriDG(ElementTriP1())
alpha = 1e-1

ib = Basis(m, e)
bb = FacetBasis(m, e)
fb = [InteriorFacetBasis(m, e, side=i) for i in [0, 1]]

@BilinearForm
def dgform(u, v, p):
    ju = p.sign1 * u
    jv = p.sign2 * v
    h = p.h
    n = p.n
    return (ju * jv) / (alpha * h)\
        - dot(grad(u), n) * jv / 2\
        - dot(grad(v), n) * ju / 2

A = asm(laplace, ib)
B = asm(dgform, fb, fb)
C = asm(dgform, bb)
b = asm(unit_load, ib)

x = solve(A + B + C, b)

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    plot(M, X, shading="gouraud", colorbar=True)
    show()
