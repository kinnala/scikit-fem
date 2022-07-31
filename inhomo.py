from skfem import *

from skfem.helpers import *
m = MeshTri().refined(5)
e = ElementTriRT1() * ElementTriP0()

basis = Basis(m, e)
fbasis = basis.boundary()

@BilinearForm
def bilinf(sigma, u, tau, v, _):
    return dot(sigma, tau) - div(sigma) * v - div(tau) * u

@LinearForm
def linf(tau, v, _):
    return 1. * v

@LinearForm
def load(tau, v, w):
    return dot(tau, w.n) * np.isclose(w.x[0], 1) * (np.sin(np.pi * w.x[1]))

A = bilinf.assemble(basis)
f = linf.assemble(basis) + load.assemble(fbasis)

x = solve(A, f)
(sig, sigbasis), (u, ubasis) = basis.split(x)
ubasis.plot(u, colorbar=True).show()
import matplotlib.pyplot as plt
plt.savefig('test.png')