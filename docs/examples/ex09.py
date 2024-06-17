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
from skfem.helpers import *
import numpy as np
from scipy.sparse import spmatrix
from scipy.sparse.linalg import LinearOperator


p = np.linspace(0, 1, 16)
m = MeshTet.init_tensor(*(p,) * 3)
basis = Basis(m, ElementTetP1())


@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


@LinearForm
def unit_load(v, _):
    return 1. * v


A = laplace.assemble(basis)
b = unit_load.assemble(basis)

I = m.interior_nodes()
x = 0. * b


Aint, bint = condense(A, b, I=I, expand=False)

preconditioners = [None, build_pc_ilu(Aint, drop_tol=1e-3)]


# try importing algebraic multigrid from external package
try:
    from pyamg import smoothed_aggregation_solver

    def build_pc_amgsa(A: spmatrix, **kwargs) -> LinearOperator:
        """AMG (smoothed aggregation) preconditioner"""
        return smoothed_aggregation_solver(A, **kwargs).aspreconditioner()

    preconditioners.append(build_pc_amgsa(Aint))

except ImportError:
    print('Skipping PyAMG')


# solve for each preconditioner
for pc in preconditioners:
    x[I] = solve(Aint, bint, solver=solver_iter_pcg(verbose=True, M=pc))


if __name__ == "__main__":
    from os.path import splitext
    from sys import argv

    # use vedo: press 5 to visualize potential, X for cutter tool
    basis.plot(x, 'vedo').show()
    # preferred: save to vtk for visualization in Paraview
    #m.save(splitext(argv[0])[0] + ".vtk", point_data{'potential': x})
