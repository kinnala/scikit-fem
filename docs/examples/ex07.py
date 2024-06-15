"""Discontinuous Galerkin method."""

from skfem import *
from skfem.helpers import grad, dot, jump
from skfem.models.poisson import laplace, unit_load

m = MeshTri.init_sqsymmetric().refined()
e = ElementTriDG(ElementTriP4())
alpha = 1e-3

ib = Basis(m, e)
bb = FacetBasis(m, e)
fb = [InteriorFacetBasis(m, e, side=i) for i in [0, 1]]


@BilinearForm
def dgform(u, v, w):
    ju, jv = jump(w, u, v)
    h = w.h
    n = w.n
    return ju * jv / (alpha * h) - dot(grad(u), n) * jv - dot(grad(v), n) * ju


A = laplace.assemble(ib)
C = dgform.assemble(bb)
b = unit_load.assemble(ib)

# calling asm(form, [...], [...]) will automatically
# assemble all combinations from the lists and sum
# the result
B = asm(dgform, fb, fb)

x = solve(A + B + C, b)

M, X = ib.refinterp(x, 4)

def visualize():
    from skfem.visuals.matplotlib import plot, draw
    ax = draw(M, boundaries_only=True)
    return plot(M, X, shading="gouraud", ax=ax, colorbar=True)

if __name__ == "__main__":
    visualize().show()
