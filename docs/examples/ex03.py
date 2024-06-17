"""Linear elastic eigenvalue problem."""

from skfem import *
from skfem.helpers import dot, ddot, sym_grad, eye, trace
import numpy as np

m1 = MeshLine(np.linspace(0, 5, 50))
m2 = MeshLine(np.linspace(0, 1, 10))
m = (m1 * m2).with_default_tags()

e1 = ElementQuad1()
e = ElementVector(e1)

basis = Basis(m, e, intorder=2)

lam = 1.
mu = 1.


def C(T):
    return 2. * mu * T + lam * eye(trace(T), T.shape[0])


@BilinearForm
def stiffness(u, v, w):
    return ddot(C(sym_grad(u)), sym_grad(v))


@BilinearForm
def mass(u, v, w):
    return dot(u, v)


K = stiffness.assemble(basis)
M = mass.assemble(basis)

D = basis.get_dofs("left")

L, x = solve(*condense(K, M, D=D),
             solver=solver_eigen_scipy_sym(k=6, sigma=0.0))

# calculate stress
y = x[:, 4]
sbasis = basis.with_element(ElementVector(e))
yi = basis.interpolate(y)
sigma = sbasis.project(C(sym_grad(yi)))

def visualize():
    from skfem.visuals.matplotlib import plot, draw
    M = MeshQuad(np.array(m.p + .5 * y[basis.nodal_dofs]), m.t)
    ax = draw(M)
    return plot(M,
                sigma[sbasis.nodal_dofs[0]],
                ax=ax,
                colorbar='$\sigma_{xx}$',
                shading='gouraud')

if __name__ == "__main__":
    visualize().show()
