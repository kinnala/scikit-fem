"""Linear elastic eigenvalue problem."""

from skfem import *
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity
import numpy as np

m1 = MeshLine(np.linspace(0, 5, 50))
m2 = MeshLine(np.linspace(0, 1, 10))
m = m1*m2

e1 = ElementQuad1()

mapping = MappingIsoparametric(m, e1)

e = ElementVectorH1(e1)

gb = Basis(m, e, mapping, 2)

K = asm(linear_elasticity(1.0, 1.0), gb)

@BilinearForm
def mass(u, v, w):
    return dot(u, v)

M = asm(mass, gb)

D = gb.find_dofs({'': m.facets_satisfying(lambda x: x[0]==0.0)})
y = gb.zeros()

I = gb.complement_dofs(D)

L, x = solve(*condense(K, M, I=I), solver=solver_eigen_scipy_sym(k=6, sigma=0.0))

y = x[:, 4]

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, show
    M = MeshQuad(np.array(m.p + y[gb.nodal_dofs]), m.t)
    draw(M)
    show()
