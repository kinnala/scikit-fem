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

_, Dleft = ib.find_dofs(lambda x,y,z: x==0.0)
u, Drightx = ib.find_dofs(lambda x,y,z: x==1.0, dofrows=[0])
_, Drightyz = ib.find_dofs(lambda x,y,z: x==1.0, dofrows=[1,2])

u[Drightx] = 0.3

I = ib.dofnum.complement_dofs(np.concatenate((Dleft, Drightx, Drightyz)))

u[I] = solve(*condense(K, 0*u, I=I, x=u))

sf = 1.0
for itr in range(3):
    m.p[itr, :] += sf*u[ib.dofnum.n_dof[itr, :]]
m.save('elasticity.vtk')
