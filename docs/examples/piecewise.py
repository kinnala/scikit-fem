"""

Piecewise-constant coefficients.

"""

from skfem import *
import numpy as np

lengths = [3, 2]

mesh = (MeshLine(np.linspace(-lengths[0], lengths[1], 6))
        * MeshLine(np.linspace(0, 1, 2)))._splitquads()
basis = InteriorBasis(mesh, ElementTriP1())
basis0 = InteriorBasis(mesh, ElementTriP0())


@bilinear_form
def conduction(u, du, v, dv, w):
    return w.w * sum(du * dv)


k = np.ones(basis.N) * lengths[1]
k[mesh.elements_satisfying(lambda x, y: x < 0)] = lengths[0]
L = asm(conduction, basis, w=basis0.interpolate(k))

cold = basis.get_dofs(lambda x, y: x == -lengths[0]).all()
hot = basis.get_dofs(lambda x, y: x == lengths[1]).all()

u = np.zeros(L.shape[0])
u[cold] = -1
u[hot] = 1
interior = basis.complement_dofs(np.concatenate([cold, hot]))
u[interior] = solve(*condense(L, np.zeros_like(u), u, interior))

u_origin = basis.interpolator(u)(np.zeros((2, 1)))[0]


if __name__ == '__main__':
    assert np.allclose(u_origin, 0)

    ax = mesh.plot(u)
    ax.axis('off')
    ax.get_figure().savefig('piecewise.png')
