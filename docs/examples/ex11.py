r"""Linear elasticity.

This example solves the linear elasticity problem using trilinear elements.  The
weak form of the linear elasticity problem is defined in
:func:`skfem.models.elasticity.linear_elasticity`.

"""

import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity, lame_parameters

m = MeshHex().refined(3)
e1 = ElementHex1()
e = ElementVectorH1(e1)
ib = Basis(m, e, MappingIsoparametric(m, e1), 3)

K = asm(linear_elasticity(*lame_parameters(1e3, 0.3)), ib)

dofs = {
    'left' : ib.get_dofs(lambda x: x[0] == 0.0),
    'right': ib.get_dofs(lambda x: x[0] == 1.0),
}

u = ib.zeros()
u[dofs['right'].nodal['u^1']] = 0.3

I = ib.complement_dofs(dofs)

u = solve(*condense(K, x=u, I=I))

sf = 1.0
m = m.translated(sf * u[ib.nodal_dofs])

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    
    m.save(splitext(argv[0])[0] + '.vtk')
