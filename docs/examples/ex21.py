from skfem import *
from skfem.models.elasticity import linear_elasticity,\
                                    lame_parameters
import numpy as np

from pathlib import Path

m = MeshTet.load(Path(__file__).with_name("beams.msh"))
e1 = ElementTetP1()
e = ElementVectorH1(e1)

ib = InteriorBasis(m, e)

K = asm(linear_elasticity(*lame_parameters(200.0e9, 0.3)), ib)

rho = 8050.0


@bilinear_form
def mass(u, du, v, dv, w):
    return rho * sum(u * v)

M = asm(mass, ib)

dofs = ib.get_dofs(m.boundaries)

D = np.concatenate((
        dofs['fixed'].nodal['u^1'],
        dofs['fixed'].nodal['u^2'],
        dofs['fixed'].nodal['u^3'],
))

I = ib.complement_dofs(D)

L, x = solve(*condense(K, M, I=I))

y = np.zeros(K.shape[0])
y[I] = x[:, 0]

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, show
    sf = 2.0
    draw(MeshTet(np.array(m.p + sf * y[ib.nodal_dofs]), m.t))
    show()
