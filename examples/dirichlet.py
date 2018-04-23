from skfem import *

m = MeshTri()
m.refine(4)

e = ElementTriP1()
map = MappingAffine(m)
basis = InteriorBasis(m, e, map, 2)

@bilinear_form
def laplace(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@linear_form
def load(v, dv, w):
    return 1.0*v

A = asm(laplace, basis)
b = asm(load, basis)

I = m.interior_nodes()

u = [1., -1.] @ m.p**2          # x**2 - y**2
u[I] = solve(*condense(A, 0.*b, u, I))

for k, exact in enumerate([1, 0, 8/45]):
    print(b @ u**k, exact)
m.plot3(u)
m.show()
