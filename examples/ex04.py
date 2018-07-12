"""
Author: kinnala

Solve a linearized contact problem between two linear elastic
bodies using the penalty method.
"""

from skfem import *
from skfem.models.elasticity import plane_strain
import numpy as np

m = MeshTri()
m.refine(3)
M = MeshLine(np.linspace(0, 1, 6))
M = M*M
M = M._splitquads()
M.translate((1.0, 0.0))

map = MappingAffine(m)
Map = MappingAffine(M)
e1 = ElementTriP1()
e = ElementVectorH1(e1)

ib = InteriorBasis(m, e, map, 2)
Ib = InteriorBasis(M, e, Map, 2)

def rule(x, y):
    return (x==1.0)

def param(x, y):
    return y

mortar = InterfaceMesh1D(m, M, rule, param, debug_plot=True)
m.show()

mb = {}
mortar_map = MappingAffine(mortar)
mb[0] = FacetBasis(mortar, e, mortar_map, 2, side=0)
mb[1] = FacetBasis(mortar, e, mortar_map, 2, side=1)

E1 = 1000.0
E2 = 1000.0

nu1 = 0.3
nu2 = 0.3

Mu1 = E1/(2.0*(1.0 + nu1))
Mu2 = E2/(2.0*(1.0 + nu2))

Lambda1 = E1*nu1/((1.0 + nu1)*(1.0 - 2.0*nu1))
Lambda2 = E2*nu2/((1.0 + nu2)*(1.0 - 2.0*nu2))

Mu = Mu1
Lambda = Lambda1

weakform1 = plane_strain(Lambda=Lambda1, Mu=Mu1)
weakform2 = plane_strain(Lambda=Lambda2, Mu=Mu2)

alpha = 1
K1 = asm(weakform1, ib)
K2 = asm(weakform2, Ib)
L = 0
for i in range(2):
    for j in range(2):
        @bilinear_form
        def bilin_penalty(u, du, v, dv, w):
            n = w[2]
            ju = (-1.0)**i*(u[0]*n[0] + u[1]*n[1])
            jv = (-1.0)**j*(v[0]*n[0] + v[1]*n[1])

            def tr(T):
                return T[0, 0] + T[1, 1]

            def C(T):
                return np.array([[2*Mu*T[0, 0] + Lambda*tr(T), 2*Mu*T[0, 1]],
                                 [2*Mu*T[1, 0], 2*Mu*T[1, 1] + Lambda*tr(T)]])

            def Eps(dw):
                return np.array([[dw[0][0], 0.5*(dw[0][1] + dw[1][0])],
                                 [0.5*(dw[1][0] + dw[0][1]), dw[1][1]]])
            mu = 0.5*(n[0]*C(Eps(du))[0, 0]*n[0] + n[0]*C(Eps(du))[0, 1]*n[1] + n[1]*C(Eps(du))[1, 0]*n[0] + n[1]*C(Eps(du))[1, 1]*n[1])
            mv = 0.5*(n[0]*C(Eps(dv))[0, 0]*n[0] + n[0]*C(Eps(dv))[0, 1]*n[1] + n[1]*C(Eps(dv))[1, 0]*n[0] + n[1]*C(Eps(dv))[1, 1]*n[1])
            h = w[1]
            return 1.0/(alpha*h)*ju*jv - mu*jv - mv*ju

        L = asm(bilin_penalty, mb[i], mb[j]) + L

@linear_form
def load(v, dv, w):
    return -50*v[1]

f1 = asm(load, ib)
f2 = np.zeros(K2.shape[0])

import scipy.sparse
K = (scipy.sparse.bmat([[K1, None],[None, K2]]) + L).tocsr()

i1 = np.arange(K1.shape[0])
i2 = np.arange(K2.shape[0]) + K1.shape[0]

_, D1 = ib.find_dofs(lambda x, y: x==0.0)
_, D2 = Ib.find_dofs(lambda x, y: x==2.0)

x = np.zeros(K.shape[0])

f = np.hstack((f1, f2))

x = np.zeros(K.shape[0])
D = np.concatenate((D1, D2 + ib.dofnum.N))
I = np.setdiff1d(np.arange(K.shape[0]), D)

x[I] = solve(*condense(K, f, I=I))

sf = 1

m.p[0, :] = m.p[0, :] + sf*x[i1][ib.dofnum.n_dof[0, :]]
m.p[1, :] = m.p[1, :] + sf*x[i1][ib.dofnum.n_dof[1, :]]

M.p[0, :] = M.p[0, :] + sf*x[i2][Ib.dofnum.n_dof[0, :]]
M.p[1, :] = M.p[1, :] + sf*x[i2][Ib.dofnum.n_dof[1, :]]

ax = m.draw()
M.draw(ax=ax)
m.show()

print(np.max(x))
