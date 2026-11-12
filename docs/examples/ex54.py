"""Distributed solution of nonlinear problem using PETSc.

This is a problem with mild nonlinearity for testing the distributed
solver.  Apart from :meth:`~skfem.mesh.Mesh.distributed`, the script is
the same as it would be in serial: the Newton iteration operates on the
local part of the solution and the norm of the update is computed using
a functional which is summed over all processes.

Run, e.g., using ``mpirun -n 4 python ex54.py``.

This example requires petsc4py, mpi4py and pymetis.

"""
from skfem import *
from skfem.autodiff import *
from skfem.autodiff.helpers import *
import numpy as np


m = MeshHex.init_tensor(
    np.linspace(0, 5, 20),
    np.linspace(0, 1, 6),
    np.linspace(0, 1, 6),
).refined().with_defaults().distributed()
basis = Basis(m, ElementHex1())


@NonlinearForm
def nonlinf(u, v, _):
    return dot((u + 1) * grad(u), grad(v)) - 1. * v


@Functional
def l2(w):
    return w['u'] ** 2


x = basis.zeros()
for itr in range(100):
    J, f = nonlinf.assemble(basis, x=x)
    dx = solve(
        *enforce(J, f, D=basis.get_dofs({'left', 'right'})),
        solver=solver_petsc(ksp_type='cg', pc_type='gamg', rtol=1e-7),
    )
    res = np.sqrt(l2.assemble(basis, u=dx))
    print(res)
    if res < 1e-6:
        break
    x += 0.95 * dx


xmax = allreduce(x.max(), op='max')
print("maximum value: {}".format(xmax))
