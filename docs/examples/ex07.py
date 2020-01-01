from skfem import *
from skfem.models.poisson import laplace, unit_load

m = MeshTri.init_sqsymmetric()
m.refine()
mapping = MappingAffine(m)
e = ElementTriDG(ElementTriP1())
alpha = 1e-1

ib = InteriorBasis(m, e)
fb = {}
fb[0] = FacetBasis(m, e, side=0)
fb[1] = FacetBasis(m, e, side=1)
bb = FacetBasis(m, e)

@bilinear_form
def bilin_bnd(u, du, v, dv, w):
    h = w.h
    n = w.n
    dudn = du[0]*n[0] + du[1]*n[1]
    dvdn = dv[0]*n[0] + dv[1]*n[1]
    return 1.0/(alpha*h)*u*v - dudn*v - u*dvdn

A = asm(laplace, ib)
b = asm(unit_load, ib)

C = asm(bilin_bnd, bb)
B = 0
for i in range(2):
    for j in range(2):
        @bilinear_form
        def bilin_int(u, du, v, dv, w):
            ju = (-1.0)**i*u
            jv = (-1.0)**j*v
            n = w.n
            mu = 0.5*(du[0]*n[0] + du[1]*n[1])
            mv = 0.5*(dv[0]*n[0] + dv[1]*n[1])
            h = w.h
            return 1.0/(alpha*h)*ju*jv - mu*jv - mv*ju

        B = asm(bilin_int, fb[i], fb[j]) + B

x = solve(A+B+C, b)

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    from skfem.visuals.matplotlib import plot, show
    plot(M, X, shading="gouraud", colorbar=True)
    show()
