from skfem import *
from skfem.models.elasticity import *

m = MeshHex()
m.refine(3)
ib = basis_elasticity(m)

K = asm(linear_elasticity(*lame_parameters(1e3, 0.3)), ib)

@boundary_clamped
def left(x, y, z):
    return x==0.0

@boundary_prescribed(0.3, 0.0, 0.0)
def right(x, y, z):
    return x==1.0

u, I = initialize(ib, left, right)

u[I] = solve(*condense(K, 0*u, I=I, x=u))

sf = 1.0
for itr in range(3):
    m.p[itr, :] += sf*u[ib.dofnum.n_dof[itr, :]]
m.save('elasticity.vtk')
