"""Distributed solution of nonlinear problem using PETSc.

This is a problem with mild nonlinearity for testing the distributed
solver.

"""
from skfem import *
from skfem.autodiff import *
from skfem.autodiff.helpers import *
import numpy as np
import petsc4py.PETSc as petsc


comm = petsc.COMM_WORLD


@Dofs.decompose(comm)
def builder():
    m = MeshHex.init_tensor(
        np.linspace(0, 5, 20),
        np.linspace(0, 1, 6),
        np.linspace(0, 1, 6),
    ).refined().with_defaults()
    dofs = Dofs(m, ElementHex1())
    return m, dofs


m, dofs = builder()


basis = Basis(m, ElementHex1())


@NonlinearForm
def nonlinf(u, v, _):
    return dot((u + 1) * grad(u), grad(v)) - 1. * v


# initialize solution
x = basis.zeros()
J, f = nonlinf.elemental(basis, x=x)
J = J.topetsc(dofs)
f = f.topetsc(dofs)
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
    J = J.topetsc(dofs)
    f = f.topetsc(dofs)

    J.zeroRowsColumns(
        dofs.l2g(basis.get_dofs({
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

    dx = dofs.loc(y)
    x += 0.95 * dx


m.save('{}_ex52.vtk'.format(comm.rank), point_data={'x': x})
print("maximum value: {}".format(np.max(x)))
