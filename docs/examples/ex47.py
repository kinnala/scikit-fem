"""Projection between two meshes using supermesh."""

import numpy as np
from skfem import *
from skfem.supermeshing import intersect, elementwise_quadrature


m1 = MeshLine(np.linspace(1, 10, 20))
m2 = MeshLine(np.logspace(0, 1, 10))
e1 = ElementLineP1()
e2 = ElementLineP1()

m12, t1, t2 = intersect(m1, m2)

bases = [
    Basis(m1, e1),
    Basis(m2, e2),
]
projbases = [
    Basis(m1, e1, quadrature=elementwise_quadrature(m1, m12, t1), elements=t1),
    Basis(m2, e2, quadrature=elementwise_quadrature(m2, m12, t2), elements=t2),
]


@BilinearForm
def mass(u, v, _):
    return u * v


P = mass.assemble(*projbases)
M = mass.assemble(projbases[1])

y1 = bases[0].project(lambda x: x[0] ** 1.6)
y2 = solve(M, P.dot(y1))


l2 = .09 * (bases[0].interpolator(y1)(np.linspace(1, 10, 100)[None])
            - bases[1].interpolator(y2)(np.linspace(1, 10, 100)[None])).sum()

if __name__ == "__main__":
    print('L2 error: {}'.format(l2))
    ax = bases[0].plot(y1, color='ko-')
    m1.draw(ax=ax, color='ko')
    m2.draw(ax=ax, color='ro')
    bases[1].plot(y2, color='ro:', ax=ax).show()
