from skfem import *
from skfem.helpers import dot
from skfem.models.elasticity import linear_elasticity
from scipy.sparse.linalg import eigsh
import numpy as np

m1 = MeshLine(np.linspace(0, 5, 50))
m2 = MeshLine(np.linspace(0, 1, 10))
m = m1*m2

e1 = ElementQuad1()

mapping = MappingIsoparametric(m, e1)

e = ElementVectorH1(e1)

gb = InteriorBasis(m, e, mapping, 2)

K = asm(linear_elasticity(1.0, 1.0), gb)

@BilinearForm
def mass(u, v, w):
    return dot(u, v)

M = asm(mass, gb)

D = gb.get_dofs(lambda x: x[0]==0.0).all()
y = np.zeros(gb.N)

I = gb.complement_dofs(D)

L, x = eigsh(K[I].T[I].T, k=6, M=M[I].T[I].T, which='SM')

y[I] = x[:, 4]

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, show
    M = MeshQuad(np.array(m.p + y[gb.nodal_dofs]), m.t)
    draw(M)
    show()
