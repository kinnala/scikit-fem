"""
Author: gdmcbain

This example demonstrates a spatially varying coefficient.

Legendre's equation in self-adjoint Sturm–Liouville form is

  {(1 - x**2) u'}' + k u = 0      (-1 < x < 1)

The eigenvalues are k = n * (n + 1) for n = 0, 1, 2, …  The
conventional normalization is u(1) = 1 and

   <u, u> = 2 / (2 * n + 1).

The x-coordinate for the spatially varying coefficient (1 - x**2) is
accessed inside the bilinear_form as w[0][0].

"""

from matplotlib.pyplot import subplots
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.special import legendre

from skfem import *
from skfem.models.poisson import mass

x = np.linspace(-1, 1)
m = MeshLine(x)
e = ElementLineP1()
basis = InteriorBasis(m, e)


@bilinear_form
def stiffness(u, du, v, dv, w):
    return dv[0] * (1 - w[0][0]**2) * du[0]


L = asm(stiffness, basis)
M = asm(mass, basis)

k, u = eigsh(L, M=M, which='SM')
u = u * np.sqrt([2 / (2 * n + 1) for n in range(len(k))]) * np.sign(u[-1, :])

fig, ax = subplots()
for n, (k, u) in enumerate(zip(k, u.T)):
    dots, = ax.plot(x, u, label=n, marker='o', linestyle='None')
    ax.plot(x, legendre(n)(x), color=dots.get_color())
    print('{:2d}  {:5.2f}'.format(n * (n + 1), k))
fig.legend()
fig.savefig('legendre')
