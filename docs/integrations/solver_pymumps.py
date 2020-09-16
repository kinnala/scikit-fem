from skfem import *
from skfem.models.poisson import *
from skfem.utils import LinearSolver
import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator
from mumps import DMumpsContext


def solver_mumps() -> LinearSolver:
    # as per https://github.com/pymumps/pypumps README.md
    def solver(A: spmatrix, b: np.ndarray) -> np.ndarray:
        ctx = DMumpsContext()
        if ctx.myid == 0:
            ctx.set_centralized_sparse(A)
            x = b.copy()
            ctx.set_rhs(x)
        ctx.run(job=6)
        ctx.destroy()
        return x

    return solver


p = np.linspace(0, 1, 16)
m = MeshTet.init_tensor(*(p,) * 3)

e = ElementTetP1()
basis = InteriorBasis(m, e)

A = asm(laplace, basis)
b = asm(unit_load, basis)

I = m.interior_nodes()

x = solve(*condense(A, b, I=I), solver=solver_mumps())

if __name__ == "__main__":
    from pathlib import Path

    m.save(Path(__file__).with_suffix(".xdmf"), {"potential": x})
