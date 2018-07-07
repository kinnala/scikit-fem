from skfem import *
import matplotlib.pyplot as plt

# Interior penalty

m = MeshTri.init_sqsymmetric()
m.refine(2)
map = MappingAffine(m)
e = ElementTriDG(ElementTriP1())

ib = InteriorBasis(m, e, map, 3)
fb = {}
fb[0] = FacetBasis(m, e, map, 3, side=0)
fb[1] = FacetBasis(m, e, map, 3, side=1)
bb = FacetBasis(m, e, map, 3)

@bilinear_form
def bilin(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@bilinear_form
def bilin_bnd(u, du, v, dv, w):
    h = w[1]
    return 1.0/h*u*v

@linear_form
def lin(v, dv, w):
    return 1.0*v

A = asm(bilin, ib)
b = asm(lin, ib)

ieps = 1e-3
beps = 10
C = beps*asm(bilin_bnd, bb)
B = 0
for i in range(2):
    for j in range(2):
        @bilinear_form
        def bilin_penalty(u, du, v, dv, w):
            ju = (-1.0)**i*u
            jv = (-1.0)**j*v
            n = w[2]
            mu = 0.5*(du[0]*n[0] + du[1]*n[1])
            mv = 0.5*(dv[0]*n[0] + dv[1]*n[1])
            h = w[1]
            return ieps/h*ju*jv - mu*jv - mv*ju

        B = asm(bilin_penalty, fb[i], fb[j]) + B

x = solve(A+B+C, b)

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    M.plot(X, smooth=True, edgecolors='')
    M.show()
