"""Linear elastic eigenvalue problem."""

from skfem import *
from skfem.helpers import dot, ddot, sym_grad
from skfem.models.elasticity import linear_elasticity, linear_stress
import numpy as np

m1 = MeshLine(np.linspace(0, 5, 50))
m2 = MeshLine(np.linspace(0, 1, 10))
m = (m1 * m2).with_boundaries(
    {
        "left": lambda x: x[0] == 0.0
    }
)

e1 = ElementQuad1()

mapping = MappingIsoparametric(m, e1)

e = ElementVector(e1)

gb = Basis(m, e, mapping, 2)

lam = 1.
mu = 1.
K = asm(linear_elasticity(lam, mu), gb)

@BilinearForm
def mass(u, v, w):
    return dot(u, v)

M = asm(mass, gb)

D = gb.get_dofs("left")
y = gb.zeros()

I = gb.complement_dofs(D)

L, x = solve(*condense(K, M, I=I),
             solver=solver_eigen_scipy_sym(k=6, sigma=0.0))

y = x[:, 4]

# calculate stress
sgb = gb.with_element(ElementVector(e))
C = linear_stress(lam, mu)
yi = gb.interpolate(y)
sigma = sgb.project(C(sym_grad(yi)))

def visualize():
    from skfem.visuals.matplotlib import plot, draw
    M = MeshQuad(np.array(m.p + .5 * y[gb.nodal_dofs]), m.t)
    ax = draw(M)
    return plot(M,
                sigma[sgb.nodal_dofs[0]],
                ax=ax,
                colorbar='$\sigma_{xx}$',
                shading='gouraud')

if __name__ == "__main__":
    visualize().show()
