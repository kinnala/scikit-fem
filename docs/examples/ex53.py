"""Distributed assembly and solve using PETSc.

This is a distributed version of ex52.  The only change to the serial
script is the call to :meth:`~skfem.mesh.Mesh.distributed`.  Then each
MPI process works on its own part of the mesh, the forms assemble into
distributed PETSc matrices and vectors, and :func:`~skfem.utils.solve`
returns the local part of the solution.  For very large meshes, use
:func:`~skfem.distributed.distribute` with a function that builds the
mesh, e.g., ``distribute(lambda: MeshTet().refined(6))``, so that the
global mesh is built only on the first process.

Run, e.g., using ``mpirun -n 4 python ex53.py``.  The solver can be
changed from the command line, e.g., ``-ksp_type gmres -ksp_monitor``.

This example requires petsc4py, mpi4py and pymetis.

"""
from skfem import *
from skfem.helpers import *


m = MeshTet().refined(6).distributed()
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

# x is the local part of the solution: reduce over the processes
xmax = allreduce(x.max(), op='max')
print(xmax)
