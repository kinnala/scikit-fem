"""Using PETSc linear solvers.

While before a conversion was possible, there is now basic built-in
support for the creation of PETSc matrices.  This allows using the
variety of linear solvers and preconditiors from PETSc.

This example requires petsc4py.

"""
from skfem import *
from skfem.helpers import *
import petsc4py.PETSc as petsc
import time


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


# build elemental matrices/vectors and convert to PETSc Mat/Vec
start = time.time()
A = bilinf.elemental(basis).topetsc()
b = linf.elemental(basis).topetsc()
print("--- %s seconds ---" % (time.time() - start))

x = A.createVecRight()

# apply dirichlet BC
A.zeroRowsColumns(
    basis.get_dofs(),
    diag=1.,
    x=x,
    b=b,
)

# create solver instance
ksp = petsc.KSP().create()
ksp.setOperators(A)
ksp.setType('cg')
ksp.getPC().setType('gamg')
ksp.setTolerances(rtol=1e-6, atol=0, divtol=1e16, max_it=400)

# solve
start = time.time()
ksp.solve(b, x)
print("--- %s seconds ---" % (time.time() - start))

# print maximum value
xmax = x.array.max()
print(xmax)
