"""Using PETSc linear solvers.

The assembled SciPy matrices can be passed to the linear solvers and
preconditioners of PETSc using :func:`~skfem.utils.solver_petsc`.  The
solver can be changed also from the command line, e.g.,
``python ex52.py -ksp_type gmres -pc_type ilu -ksp_monitor``.

See ex53 for running the same script in parallel using MPI: the only
change is to call :meth:`~skfem.mesh.Mesh.distributed`.

This example requires petsc4py.

"""
from skfem import *
from skfem.helpers import *


# this mesh has 230 945 vertices
m = MeshTet().refined(6)
basis = Basis(m, ElementTetP1())


@BilinearForm
def bilinf(u, v, _):
    return dot(grad(u), grad(v))


@LinearForm
def linf(v, w):
    return 3 * np.pi ** 2 * (np.sin(np.pi * w.x[0])
                             * np.sin(np.pi * w.x[1])
                             * np.sin(np.pi * w.x[2])) * v


A = bilinf.assemble(basis)
b = linf.assemble(basis)

x = solve(
    *condense(A, b, D=basis.get_dofs()),
    solver=solver_petsc(ksp_type='cg', pc_type='gamg', rtol=1e-6),
)

# allreduce is a no-op in serial; see ex53
xmax = allreduce(x.max(), op='max')
print(xmax)
