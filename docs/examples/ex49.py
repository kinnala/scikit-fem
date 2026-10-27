"""Projection between two meshes using supermesh in 2D (or 3D)."""

import numpy as np
from skfem import *
from skfem.supermeshing import intersect, elementwise_quadrature


# save projection errors here
L2s = []


for dim in [2, 3]:

    if dim == 2:
        # 2D version 
        m1 = MeshTri.init_tensor(np.linspace(0, 1, 4),
                                 np.linspace(0, 1, 4))
        m2 = MeshQuad().refined(2)
        e1 = ElementTriP2()
        e2 = ElementQuad2()

    elif dim == 3:
        # 3D version
        m1 = MeshTet.init_tensor(np.linspace(0, 1, 4),
                                 np.linspace(0, 1, 4),
                                 np.linspace(0, 1, 4))
        m2 = MeshTet.init_tensor(np.linspace(0, 1, 6),
                                 np.linspace(0, 1, 6),
                                 np.linspace(0, 1, 6))
        e1 = ElementTetP2()
        e2 = ElementTetP1()

    m12, t1, t2 = intersect(m1, m2)

    b1 = Basis(m1, e1)
    b2 = Basis(m2, e2)

    pb1 = Basis(m1, e1, quadrature=elementwise_quadrature(m1, m12, t1, intorder=4), elements=t1)
    pb2 = Basis(m2, e2, quadrature=elementwise_quadrature(m2, m12, t2, intorder=4), elements=t2)


    @BilinearForm
    def mass(u, v, _):
        return u * v


    P = mass.assemble(pb1, pb2)
    M = mass.assemble(pb2)

    fun = lambda x: (x[0] * x[1]) ** 1.6
    y1 = b1.project(fun)
    y2 = solve(M, P.dot(y1))


    @Functional
    def error(w):
        return (fun(w.x) - w['y2']) ** 2


    L2 = np.sqrt(error.assemble(b2, y2=y2))
    L2s.append(L2)

    if __name__ == "__main__":
        print('L2 error: {}'.format(L2))
        if m1.dim() == 2:
            ax = b1.plot(y1, colorbar=True, shading='gouraud', nrefs=3)
            m1.draw(ax=ax, color='ko-')
            ax = b2.plot(y2, colorbar=True, shading='gouraud')
            m2.draw(ax=ax, color='ro-').show()
