"""Distributed solution of nonlinear problem using PETSc."""
from skfem import *
from skfem.autodiff import *
from skfem.autodiff.helpers import *
import numpy as np
import petsc4py.PETSc as petsc


comm = petsc.COMM_WORLD
m = MeshHex.init_tensor(
    np.linspace(0, 5, 20),
    np.linspace(0, 1, 6),
    np.linspace(0, 1, 6),
).refined(3).with_defaults().distributed(comm)
basis = Basis(m, ElementHex1(), intorder=3)


@NonlinearForm
def nonlinf(u, v, _):
    return dot((u + 1) * grad(u), grad(v)) - 1. * v


# initialize solution
x = basis.zeros()
J, f = nonlinf.elemental(basis, x=x)
J = J.topetsc(basis.dofs)
f = f.topetsc(basis.dofs)
y = J.createVecRight()

# create solver instance
ksp = petsc.KSP().create()
ksp.setOperators(J)
ksp.setType('cg')
ksp.getPC().setType('gamg')
ksp.setTolerances(rtol=1e-7, atol=0, divtol=1e16, max_it=400)


for itr in range(100):
    x_prev = x.copy()
    J, f = nonlinf.elemental(basis, x=x)
    J = J.topetsc(basis.dofs)
    f = f.topetsc(basis.dofs)

    J.zeroRowsColumns(
        basis.dofs.l2g(basis.get_dofs({
            'left',
            'right',
        })),
        diag=1.,
        x=y,
        b=f,
    )

    ksp.setOperators(J)

    ksp.solve(f, y)

    #print(ksp.its)

    res = y.norm()
    if res < 1e-6:
        break
    if comm.rank == 0:
        print(res)

    dx = basis.dofs.loc(y)
    x += 0.95 * dx


m.save('{}_ex52.vtk'.format(comm.rank), point_data={'x': x})
print("maximum value: {}".format(np.max(x)))
