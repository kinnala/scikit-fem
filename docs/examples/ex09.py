from skfem import *
from skfem.models.poisson import *
import numpy as np

p = np.linspace(0, 1, 16)
m = MeshTet.init_tensor(*(p,)*3)

e = ElementTetP1()
basis = InteriorBasis(m, e)

A = asm(laplace, basis)
b = asm(unit_load, basis)

I = m.interior_nodes()

x = 0*b

if __name__ == "__main__":
    verbose = True
else:
    verbose = False
# run conjugate gradient with the default preconditioner
Aint, bint = condense(A, b, I=I, expand=False)
x[I] = solve(Aint, bint, solver=solver_iter_pcg(verbose=verbose))

# run conjugate gradient with the incomplete LU preconditioner
x[I] = solve(Aint, bint,
             solver=solver_iter_pcg(M=build_pc_ilu(Aint), verbose=verbose))

if verbose:
    from os.path import splitext
    from sys import argv

    m.save(splitext(argv[0])[0] + ".vtk", {'potential': x})
