"""Projection between two meshes using supermesh."""

import numpy as np
from skfem import *
from skfem.supermeshing import intersect, elementwise_quadrature


m1 = MeshTri.init_tensor(np.linspace(0, 1, 4),
                         np.linspace(0, 1, 4))
m2 = MeshQuad().refined(2)
e1 = ElementTriP2()
e2 = ElementQuad2()

m12, t1, t2 = intersect(m1, m2)

bases = [
    Basis(m1, e1),
    Basis(m2, e2),
]
projbases = [
    Basis(m1, e1, quadrature=elementwise_quadrature(m1, m12, t1, intorder=4), elements=t1),
    Basis(m2, e2, quadrature=elementwise_quadrature(m2, m12, t2, intorder=4), elements=t2),
]


@BilinearForm
def mass(u, v, _):
    return u * v


P = mass.assemble(*projbases)
M = mass.assemble(projbases[1])

y1 = bases[0].project(lambda x: x[0] ** 1.6)
y2 = solve(M, P.dot(y1))

xs = np.linspace(0, 1, 100)
xs = np.vstack((xs, xs))
l2 = (bases[0].interpolator(y1)(xs)
      - bases[1].interpolator(y2)(xs)).sum()

if __name__ == "__main__":
    print('L2 error: {}'.format(l2))
    ax = bases[0].plot(y1, colorbar=True, shading='gouraud')
    m1.draw(ax=ax, color='ko')
    ax = bases[1].plot(y2, colorbar=True, shading='gouraud')
    m2.draw(ax=ax, color='ro').show()
