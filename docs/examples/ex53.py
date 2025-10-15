"""Distributed assembly and solve using PETSc.

This is a distributed version of ex52.

"""
from skfem import *
from skfem.helpers import *
import petsc4py.PETSc as petsc
import time

# this mesh has 230 945 vertices
# split the mesh to ranks using pymetis
comm = petsc.COMM_WORLD
m = MeshTet().refined(6).with_defaults().distributed(comm)
basis = Basis(m, ElementTetP1())
# m = MeshTri().refined(6).with_defaults().distributed(comm)
# basis = Basis(m, ElementTriP1())


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

# Dofs object contains local-to-global DOF mapping
# it is required for distributed assembly
A = bilinf.elemental(basis).topetsc(basis.dofs)
b = linf.elemental(basis).topetsc(basis.dofs)
print("--- %s seconds ---" % (time.time() - start))

x = A.createVecRight()

# apply dirichlet BC
# NOTE: cannot use basis.get_dofs() for full dirichlet BC since local
# boundary is different than global boundary: must use tags 'left',
# 'right', ...
A.zeroRowsColumns(
    basis.dofs.l2g(basis.get_dofs({
        'left',
        'right',
        'bottom',
        'top',
        'front',
        'back',
    })),  # local-to-global DOFs
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
_, xmax = x.max()
print(xmax)
