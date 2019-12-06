from skfem import *
from skfem.models.poisson import *
import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

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

Aint, bint = condense(A, b, I=I, expand=False)

preconditioners = [None, build_pc_ilu(Aint)]

try:
    from pyamg import smoothed_aggregation_solver

    def build_pc_amgsa(A: spmatrix, **kwargs) -> LinearOperator:
        """AMG (smoothed aggregation) precondtioner"""
        return smoothed_aggregation_solver(A, **kwargs).aspreconditioner()

    preconditioners.append(build_pc_amgsa(Aint))
    
except ImportError:
    print('Skipping PyAMG')

try:
    import pyamgcl

    def build_pc_amgcl(A: spmatrix, **kwargs) -> LinearOperator:
        """AMG preconditioner"""

        if hasattr(pyamgcl, 'amgcl'): # v. 1.3.99+
            return pyamgcl.amgcl(A, **kwargs)
        else:
            return pyamgcl.amg(A, **kwargs)

    preconditioners.append(build_pc_amgcl(Aint))

except ImportError:
    print('Skipping pyamgcl')

for pc in preconditioners:
    x[I] = solve(Aint, bint, solver=solver_iter_pcg(M=pc, verbose=verbose))




if verbose:
    from os.path import splitext
    from sys import argv

    m.save(splitext(argv[0])[0] + ".vtk", {'potential': x})
