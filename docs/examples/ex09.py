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
# run conjugate gradient with the default preconditioner
Aint, bint = condense(A, b, I=I, expand=False)
x[I] = solve(Aint, bint, solver=solver_iter_pcg(verbose=verbose))

# run conjugate gradient with the incomplete LU preconditioner
x[I] = solve(Aint, bint,
             solver=solver_iter_pcg(M=build_pc_ilu(Aint), verbose=verbose))

try:
    from pyamg import smoothed_aggregation_solver

    def solver_amg_sa(**kwargs) -> 'LinearSolver':

        if kwargs.pop('verbose', False):

            def callback(x):
                print(np.linalg.norm(x))

            kwargs['callback'] = callback

        def solver(A, b):
            return smoothed_aggregation_solver(A).solve(b, **kwargs)

        return solver

    x[I] = solve(Aint, bint, solver=solver_amg_sa(verbose=verbose, accel='cg'))
    print('CG + AMG(SA) from PyAMG converged to default tol')

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
    

    x[I] = solve(Aint, bint, solver=solver_iter_pcg(M=build_pc_amgcl(Aint),
                                                    verbose=verbose))
except ImportError:
    print('Skipping pyamgcl')

if verbose:
    from os.path import splitext
    from sys import argv

    m.save(splitext(argv[0])[0] + ".vtk", {'potential': x})
