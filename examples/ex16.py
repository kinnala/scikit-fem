"""
Author: kinnala

Solving example 9 with iterative methods.
"""
from skfem import *
from skfem.models.poisson import *

m = MeshHex()
m.refine(4)

e = ElementHex1()
map = MappingIsoparametric(m, e)
basis = InteriorBasis(m, e, map, 3)

A = asm(laplace, basis)
b = asm(unit_load, basis)

I = m.interior_nodes()

x = 0*b

if __name__ == "__main__":
    verbose = True
else:
    verbose = False
# run conjugate gradient with the default preconditioner
x[I] = solve(*condense(A, b, I=I), solver=solver_iter_pcg(verbose=verbose))

# run conjugate gradient with the incomplete LU preconditioner
Aint, bint = condense(A, b, I=I)
x[I] = solve(Aint, bint, solver=solver_iter_pcg(pc=build_pc_ilu(Aint), verbose=verbose))
