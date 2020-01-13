from skfem import *
from skfem.models.elasticity import linear_elasticity
from skfem.visuals.matplotlib import *
import numpy as np

# create meshes
m = MeshTri.init_symmetric()
#m = MeshTri()
m.refine(4)
M = MeshLine(np.linspace(0, 1, 6))
M = M*M
#M = MeshTri.init_symmetric()
#M = MeshTri()
M.refine(1)
M.translate((1.0, 0.0))
M = M._splitquads()

#m.refine(3)
#M.refine(3)

e1 = ElementTriP1()
e = ElementVectorH1(e1)

E1 = ElementTriP1()
E = ElementVectorH1(E1)

ib = InteriorBasis(m, e)
Ib = InteriorBasis(M, E)

mortar = MeshMortar.init_1D(m, M,
                            m.facets_satisfying(lambda x: x[0] == 1.0),
                            M.facets_satisfying(lambda x: x[0] == 1.0),
                            np.array([0.0, 1.0]))

mb = [
    MortarBasis(m, e, mapping = mortar[0], intorder=2),
    MortarBasis(M, E, mapping = mortar[1], intorder=2)
]

E1 = 1000.0
E2 = 1000.0

nu1 = 0.3
nu2 = 0.3

Mu1 = E1 / (2. * (1. + nu1))
Mu2 = E2 / (2. * (1. + nu2))

Lambda1 = E1 * nu1 / ((1. + nu1) * (1. - 2. * nu1))
Lambda2 = E2 * nu2 / ((1. + nu2) * (1. - 2. * nu2))

Mu = Mu1
Lambda = Lambda1

weakform1 = linear_elasticity(Lambda=Lambda, Mu=Mu)
weakform2 = linear_elasticity(Lambda=Lambda, Mu=Mu)


alpha = 1
K1 = asm(weakform1, ib)
K2 = asm(weakform2, Ib)
L = [[None,None],[None,None]]
for i in range(2):
    for j in range(2):
        @bilinear_form
        def bilin_penalty(u, du, v, dv, w):
            n = w.n
            ju = (-1.) ** i * (u[0] * n[0] + u[1] * n[1])
            jv = (-1.) ** j * (v[0] * n[0] + v[1] * n[1])

            def tr(T):
                return T[0, 0] + T[1, 1]

            def C(T):
                return np.array([[2*Mu*T[0, 0] + Lambda*tr(T), 2*Mu*T[0, 1]],
                                 [2*Mu*T[1, 0], 2*Mu*T[1, 1] + Lambda*tr(T)]])

            def Eps(dw):
                return np.array([[dw[0][0], 0.5*(dw[0][1] + dw[1][0])],
                                 [0.5*(dw[1][0] + dw[0][1]), dw[1][1]]])
            mu = 0.5*(n[0]*C(Eps(du))[0, 0]*n[0] +\
                      n[0]*C(Eps(du))[0, 1]*n[1] +\
                      n[1]*C(Eps(du))[1, 0]*n[0] +\
                      n[1]*C(Eps(du))[1, 1]*n[1])
            mv = 0.5*(n[0]*C(Eps(dv))[0, 0]*n[0] +\
                      n[0]*C(Eps(dv))[0, 1]*n[1] +\
                      n[1]*C(Eps(dv))[1, 0]*n[0] +\
                      n[1]*C(Eps(dv))[1, 1]*n[1])
            h = w.h
            return 1.0/(alpha*h)*ju*jv - mu*jv - mv*ju

        L[i][j] = asm(bilin_penalty, mb[i], mb[j])

import pdb; pdb.set_trace()

@linear_form
def load(v, dv, w):
    return 50*v[1]

f1 = asm(load, ib)
f2 = np.zeros(K2.shape[0])

import scipy.sparse
K = (scipy.sparse.bmat([[K1 + L[0][0], L[1][0]],[L[0][1], K2 + L[1][1]]])).tocsr()

i1 = np.arange(K1.shape[0])
i2 = np.arange(K2.shape[0]) + K1.shape[0]

D1 = ib.get_dofs(lambda x: x[0]==0.0).all()
D2 = Ib.get_dofs(lambda x: x[0]==2.0).all()

x = np.zeros(K.shape[0])

f = np.hstack((f1, f2))

x = np.zeros(K.shape[0])
D = np.concatenate((D1, D2 + ib.N))
I = np.setdiff1d(np.arange(K.shape[0]), D)

x = solve(*condense(K, f, I=I))

sf = 1

m.p[0, :] = m.p[0, :] + sf*x[i1][ib.nodal_dofs[0, :]]
m.p[1, :] = m.p[1, :] + sf*x[i1][ib.nodal_dofs[1, :]]

M.p[0, :] = M.p[0, :] + sf*x[i2][Ib.nodal_dofs[0, :]]
M.p[1, :] = M.p[1, :] + sf*x[i2][Ib.nodal_dofs[1, :]]

if __name__ == "__main__":

    from skfem.visuals.matplotlib import draw, show

    ax = draw(m)
    draw(M, ax=ax)
    show()
