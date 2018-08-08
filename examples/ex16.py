"""
Author: gdmcbain

This example demonstrates a spatially varying coefficient.

Legendre's equation in self-adjoint Sturm–Liouville form is

  {(1 - x**2) u'}' + k u = 0      (-1 < x < 1)

The eigenvalues are k = n * (n + 1) for n = 0, 1, 2, ...  The
conventional normalization is u(1) = 1.

The x-coordinate for the spatially varying coefficient (1 - x**2) is
accessed inside the bilinear_form as w[0][0].

"""

from matplotlib.pyplot import subplots, show
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
    return (1 - w[0][0]**2) * du[0] * dv[0]

L = asm(stiffness, basis)
M = asm(mass, basis)

ks, u = eigsh(L, M=M, sigma=0.)
u /= u[-1, :]

if __name__ == "__main__":
    fig, ax = subplots()
    for n, (k, u) in enumerate(zip(ks, u.T)):
        dots, = ax.plot(x, u, label=n, marker='o', linestyle='None')
        ax.plot(x, legendre(n)(x), color=dots.get_color())
        print('{:2d}  {:5.2f}'.format(n * (n + 1), k))

    ax.legend()
    show()
