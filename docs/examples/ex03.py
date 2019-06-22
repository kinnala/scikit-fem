from skfem import *
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

@bilinear_form
def mass(u, du, v, dv, w):
    return u[0] * v[0] + u[1] * v[1]

M = asm(mass, gb)

D = gb.get_dofs(lambda x: x[0]==0.0).all()
y = np.zeros(gb.N)

I = gb.complement_dofs(D)

L, x = eigsh(K[I].T[I].T, k=6, M=M[I].T[I].T, which='SM')

y[I] = x[:, 4]

if __name__ == "__main__":
    MeshQuad(np.array([m.p[0, :] + y[gb.nodal_dofs[0, :]],
                       m.p[1, :] + y[gb.nodal_dofs[1, :]]]), m.t).draw()
    m.show()
