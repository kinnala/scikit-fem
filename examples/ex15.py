"""
Author: gdmcbain

Another simple modification of examples/ex1.py, this time showing how to pass
the x keyword-argument to condense to impose inhomogeneous Dirichlet
conditions. The forcing term is suppressed for simplicity. The boundary values
are set as the real part x**2 - y**2 of an analytic complex function z**2 which
is harmonic and so that is the exact solution through the domain.

This is checked quantitatively by computing the integral of the squared
magnitude of the gradient, by evaluating the quadratic form associated with the
laplacian at the solution; the exact value is 8/3.
"""

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

u = (([1., 1.j] @ m.p) ** 2).real          # x**2 - y**2
u[I] = solve(*condense(A, 0.*b, u, I))

print('||grad u||**2 = {:.4f} (exact = 8/3 = {:.4f})'.format(u @ A @ u, 8/3))
m.plot3(u)
m.show()
