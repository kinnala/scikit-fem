r"""Preconditioned conjugate gradient for 3-D Poisson.

.. note::

   This example will make use of the external packages `PyAMG <https://pypi.org/project/pyamg/>`_ or `pyamgcl <https://pypi.org/project/pyamgcl/>`_, if installed.

Whereas most of the examples thus far have used direct linear solvers, this is
not appropriate for larger problems, which includes most of those posed in
three dimensions.

Direct methods fail to scale beyond a certain size, typically of the order of a
few millions of unknowns, due to their intrinsic memory requirements and shear
computational cost. This makes preconditioned iterative methods the only viable
approach for solution of large scale problems.  (Demidov 2019)

scikit-fem provides access to simple preconditioners (diagonal and
incomplete-LU) from SciPy, but it's also easy to connect others from external
packages, e.g.  PyAMG or AMGCL for algebraic multigrid.

The combination of a Krylov subspace method with algebraic multigrid (AMG) as a
preconditioner is considered to be one of the most effective choices for
solution of such systems. (Demidov 2019)

These four preconditioners are demonstrated with a conjugate gradient solver for
a simple Poisson problem,

.. math::
    \begin{aligned}
        -\Delta u &= 1, && \text{in $\Omega$},\\
        u &= 0, && \text{on $\partial \Omega$},
    \end{aligned}

where :math:`\Omega = (0,1)^3`.

*  Demidov, D. (2019). AMGCL: an efficient, flexible, and extensible algebraic multigrid implementation. `arXiv:1811.05704 <https://arxiv.org/abs/1811.05704>`_

"""

from skfem import *
from skfem.models.poisson import *
import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator

p = np.linspace(0, 1, 16)
m = MeshTet.init_tensor(*(p,) * 3)

e = ElementTetP1()
basis = Basis(m, e)

A = asm(laplace, basis)
b = asm(unit_load, basis)

I = m.interior_nodes()

x = 0. * b

if __name__ == "__main__":
    verbose = True
else:
    verbose = False

Aint, bint, _ = condense(A, b, I=I)

preconditioners = [None, build_pc_ilu(Aint, drop_tol=1e-3)]

try:
    from pyamg import smoothed_aggregation_solver

    def build_pc_amgsa(A: spmatrix, **kwargs) -> LinearOperator:
        """AMG (smoothed aggregation) preconditioner"""
        return smoothed_aggregation_solver(A, **kwargs).aspreconditioner()

    preconditioners.append(build_pc_amgsa(Aint))

except ImportError:
    print('Skipping PyAMG')

try:
    import pyamgcl

    def build_pc_amgcl(A: spmatrix, **kwargs) -> LinearOperator:
        """AMG preconditioner"""

        if hasattr(pyamgcl, 'amgcl'):  # v. 1.3.99+
            return pyamgcl.amgcl(A, **kwargs)
        else:
            return pyamgcl.amg(A, **kwargs)

    preconditioners.append(build_pc_amgcl(Aint))

except ImportError:
    print('Skipping pyamgcl')

for pc in preconditioners:
    x[I] = solve(Aint, bint, solver=solver_iter_pcg(verbose=verbose, M=pc))


if verbose:
    from os.path import splitext
    from sys import argv

    m.draw('vedo', point_data={'potential': x}).show()
    #m.save(splitext(argv[0])[0] + ".vtk", {'potential': x})
