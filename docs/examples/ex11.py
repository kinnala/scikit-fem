import numpy as np
from skfem import *
from skfem.models.elasticity import linear_elasticity, \
        lame_parameters

m = MeshHex()
m.refine(3)
e1 = ElementHex1()
e = ElementVectorH1(e1)
map = MappingIsoparametric(m, e1)
ib = InteriorBasis(m, e, map, 3)

K = asm(linear_elasticity(*lame_parameters(1e3, 0.3)), ib)

dofs = {
    'left' : ib.get_dofs(lambda x: x[0]==0.0),
    'right': ib.get_dofs(lambda x: x[0]==1.0),
    }

u = np.zeros(K.shape[0])
u[dofs['right'].nodal['u^1']] = 0.3

I = ib.complement_dofs(dofs)

u[I] = solve(*condense(K, 0*u, I=I, x=u))

sf = 1.0
for itr in range(3):
    m.p[itr, :] += sf*u[ib.nodal_dofs[itr, :]]

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    
    m.save(splitext(argv[0])[0] + '.vtk')
