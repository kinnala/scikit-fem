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
    return (1 - w.x[0]**2) * du[0] * dv[0]

L = asm(stiffness, basis)
M = asm(mass, basis)

ks, u = eigsh(L, M=M, sigma=0.)
u /= u[basis.get_dofs().all()[-1], :]

if __name__ == "__main__":
    fig, ax = subplots()
    for n, (k, u) in enumerate(zip(ks, u.T)):
        dots, = ax.plot(x, u, label=n, marker='o', linestyle='None')
        ax.plot(x, legendre(n)(x), color=dots.get_color())
        print('{:2d}  {:5.2f}'.format(n * (n + 1), k))

    ax.legend()
    show()
