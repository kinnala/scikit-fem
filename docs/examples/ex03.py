"""Linear elastic eigenvalue problem."""

from skfem import *
from skfem.helpers import dot, ddot, sym_grad, eye, trace
import numpy as np

m1 = MeshLine(np.linspace(0, 5, 50))
m2 = MeshLine(np.linspace(0, 1, 10))
m = (m1 * m2).with_defaults()

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
    from skfem.visuals.web import plot
    # sigma_xx on the deformed mesh; the viewer has a menu for the other
    # components
    return plot(basis,
                {'sigma': (sbasis, sigma), 'u': y},
                component=0,
                deform='u',
                deform_scale=.5)

if __name__ == "__main__":
    visualize().show()
