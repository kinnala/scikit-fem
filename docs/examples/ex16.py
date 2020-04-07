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
u /= u[basis.get_dofs().all()[-1], :]

if __name__ == "__main__":
    fig, ax = subplots()
    for n, (k, u) in enumerate(zip(ks, u.T)):
        dots, = ax.plot(x, u[basis.nodal_dofs[0]],
                        label=n, marker='o', linestyle='None')
        ax.plot(x, legendre(n)(x), color=dots.get_color())
        print('{:2d}  {:5.2f}'.format(n * (n + 1), k))

    ax.legend()
    show()
