from skfem import *
from skfem.models.elasticity import linear_elasticity,\
                                    lame_parameters
import numpy as np

from pathlib import Path

m = MeshTet.load(Path(__file__).with_name("beams.msh"))
e1 = ElementTetP2()
e = ElementVectorH1(e1)

ib = InteriorBasis(m, e)

K = asm(linear_elasticity(*lame_parameters(200.0e9, 0.3)), ib)

rho = 8050.0


@BilinearForm
def mass(u, v, w):
    from skfem.helpers import dot
    return dot(rho * u, v)

M = asm(mass, ib)

dofs = ib.get_dofs(m.boundaries)

D = np.concatenate((
        dofs['fixed'].nodal['u^1'],
        dofs['fixed'].nodal['u^2'],
        dofs['fixed'].nodal['u^3'],
))

L, x = solve(*condense(K, M, D=ib.find_dofs()['fixed']))

if __name__ == "__main__":
    from skfem.visuals.matplotlib import draw, show
    sf = 2.0
    draw(MeshTet(np.array(m.p + sf * x[ib.nodal_dofs, 0]), m.t))
    show()
