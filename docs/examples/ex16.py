r"""Spatially varying coefficient.

This example demonstrates a spatially varying coefficient.

Legendre's equation in self-adjoint Sturm–Liouville form is

.. math::
   
  ((1 - x^2) u')' + k u = 0, \quad (-1 < x < 1)
The eigenvalues are :math:`k = n (n + 1)` for :math:`n = 0, 1, 2, \dots`  The
conventional normalization is :math:`u(1) = 1`.

The x-coordinate for the spatially varying coefficient :math:`1 - x^2` is
accessed inside the bilinear form as `w.x[0]`.

"""

from matplotlib.pyplot import subplots, show
import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.special import legendre

from skfem import *
from skfem.helpers import d, dot
from skfem.models.poisson import mass

x = np.linspace(-1, 1)
m = MeshLine(x)
e = ElementLineP1()
basis = InteriorBasis(m, e)


@BilinearForm
def stiffness(u, v, w):
    return (1 - w.x[0]**2) * dot(d(u), d(v))


L = asm(stiffness, basis)
M = asm(mass, basis)

ks, u = eigsh(L, M=M, sigma=0.)
u /= u[basis.find_dofs()['all'].nodal['u'][-1], :]

if __name__ == "__main__":
    fig, ax = subplots()
    for n, (k, u) in enumerate(zip(ks, u.T)):
        dots, = ax.plot(x, u[basis.nodal_dofs[0]],
                        label=n, marker='o', linestyle='None')
        ax.plot(x, legendre(n)(x), color=dots.get_color())
        print('{:2d}  {:5.2f}'.format(n * (n + 1), k))

    ax.legend()
    show()
