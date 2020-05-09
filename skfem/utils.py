"""This module contains utility functions such as convenient access to
SciPy linear solvers."""

import warnings
from typing import Optional, Union, Tuple,\
    Callable, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as spg
import scipy.sparse.linalg as spl
from numpy import ndarray
from scipy.sparse import spmatrix

from skfem.assembly import asm, bilinear_form, linear_form, Dofs
from skfem.assembly.basis import Basis
from skfem.element import ElementVectorH1


# custom types for describing input and output values


LinearSolver = Callable[[spmatrix, ndarray], ndarray]
EigenSolver = Callable[[spmatrix, spmatrix], Tuple[ndarray, ndarray]]
CondensedSystem = Union[spmatrix,
                        Tuple[spmatrix, ndarray],
                        Tuple[spmatrix, spmatrix],
                        Tuple[spmatrix, ndarray, ndarray],
                        Tuple[spmatrix, ndarray, ndarray, ndarray],
                        Tuple[spmatrix, spmatrix, ndarray, ndarray]]
DofsCollection = Union[ndarray, Dofs, Dict[str, Dofs]]


# preconditioners, e.g. for :func:`skfem.utils.solver_iter_krylov`


def build_pc_ilu(A: spmatrix,
                 drop_tol: Optional[float] = 1e-4,
                 fill_factor: Optional[float] = 20) -> spl.LinearOperator:
    """Incomplete LU preconditioner."""
    P = spl.spilu(A.tocsc(), drop_tol=drop_tol, fill_factor=fill_factor)
    M = spl.LinearOperator(A.shape, matvec=P.solve)
    return M


def build_pc_diag(A: spmatrix) -> spmatrix:
    """Diagonal preconditioner."""
    return sp.spdiags(1.0/A.diagonal(), 0, A.shape[0], A.shape[0])


# solvers for :func:`skfem.utils.solve`


def solver_eigen_scipy(**kwargs) -> EigenSolver:
    """Solve generalized eigenproblem using SciPy (ARPACK).

    Returns
    -------
    EigenSolver
        A solver function that can be passed to :func:`solve`.

    """
    params = {
        'sigma': 10,
        'k': 5,
        'mode': 'normal',
    }
    params.update(kwargs)
    def solver(K, M, **solve_time_kwargs):
        params.update(solve_time_kwargs)
        from scipy.sparse.linalg import eigsh
        return eigsh(K, M=M, **params)
    return solver


def solver_direct_scipy(**kwargs) -> LinearSolver:
    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        return spl.spsolve(A, b, **kwargs)
    return solver


def solver_iter_krylov(krylov: Optional[LinearSolver] = spl.cg,
                       verbose: Optional[bool] = False,
                       **kwargs) -> LinearSolver:
    """Krylov-subspace iterative linear solver.

    Parameters
    ----------
    krylov
        A Krylov iterative linear solver, like, and by default,
        :func:`scipy.sparse.linalg.cg`
    verbose
        If True, print the norm of the iterate.

    Any remaining keyword arguments are passed on to the solver, in particular
    tol and atol, the tolerances, maxiter, and M, the preconditioner.  If the
    last is omitted, a diagonal preconditioner is supplied using
    :func:`skfem.utils.build_pc_diag`.

    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.

    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    def solver(A, b, **solve_time_kwargs):
        kwargs.update(solve_time_kwargs)
        if 'M' not in kwargs:
            kwargs['M'] = build_pc_diag(A)
        sol, info = krylov(A, b, **{'callback': callback, **kwargs})
        if info > 0:
            warnings.warn("Convergence not achieved!")
        elif info == 0 and verbose:
            print(f"{krylov.__name__} converged to "
                  + f"tol={kwargs.get('tol', 'default')} and "
                  + f"atol={kwargs.get('atol', 'default')}")
        return sol

    return solver


def solver_iter_pcg(**kwargs) -> LinearSolver:
    """Conjugate gradient solver, specialized from solver_iter_krylov"""
    return solver_iter_krylov(**kwargs)


# solve and condense


def solve(A: spmatrix,
          b: Union[ndarray, spmatrix],
          x: Optional[ndarray] = None,
          I: Optional[ndarray] = None,
          solver: Optional[Union[LinearSolver, EigenSolver]] = None,
          **kwargs) -> ndarray:
    """Solve a linear system or a generalized eigenvalue problem.

    The remaining keyword arguments are passed to the solver.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector or the mass matrix of a generalized
        eigenvalue problem.
    solver
        Choose one of the following solvers:
        :func:`skfem.utils.solver_direct_scipy` (default),
        :func:`skfem.utils.solver_eigen_scipy` (default),
        :func:`skfem.utils.solver_iter_pcg`,
        :func:`skfem.utils.solver_iter_krylov`.

    """
    if solver is None:
        if isinstance(b, spmatrix):
            solver = solver_eigen_scipy(**kwargs)
        elif isinstance(b, ndarray):
            solver = solver_direct_scipy(**kwargs)

    if x is not None and I is not None:
        if isinstance(b, spmatrix):
            L, X = solver(A, b, **kwargs)
            y = np.tile(x.copy()[:, None], (1, X.shape[1]))
            y[I] = X
            return L, y
        else:
            y = x.copy()
            y[I] = solver(A, b, **kwargs)
            return y
    else:
        return solver(A, b, **kwargs)


def _flatten_dofs(S: DofsCollection) -> ndarray:
    if S is None:
        return None
    else:
        if isinstance(S, ndarray):
            return S
        elif isinstance(S, Dofs):
            return S.all()
        elif isinstance(S, dict):
            return np.unique(np.concatenate([S[key].all() for key in S]))
        raise NotImplementedError("Unable to flatten the given set of DOF's.")


def condense(A: spmatrix,
             b: Union[ndarray, spmatrix] = None,
             x: ndarray = None,
             I: DofsCollection = None,
             D: DofsCollection = None,
             expand: bool = True) -> CondensedSystem:
    """Eliminate degrees-of-freedom from a linear system.

    Supports also generalized eigenvalue problems.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector or the mass matrix for generalized
        eigenvalue problems.
    x
        The values of the condensed degrees-of-freedom. If not given, assumed
        to be zero.
    I
        The set of degree-of-freedom indices to keep.
    D
        The set of degree-of-freedom indices to dismiss.
    expand
        If `True` (default), returns also `x` and `I`. As a consequence,
        :func:`skfem.utils.solve` will expand the solution vector
        automatically. Otherwise, r

    Returns
    -------
    CondensedSystem
        The condensed linear system (and optionally information about
        the boundary values).
    """
    D = _flatten_dofs(D)
    I = _flatten_dofs(I)

    if x is None:
        x = np.zeros(A.shape[0])

    if I is None and D is None:
        raise Exception("Either I or D must be given!")
    elif I is None and D is not None:
        I = np.setdiff1d(np.arange(A.shape[0]), D)
    elif D is None and I is not None:
        D = np.setdiff1d(np.arange(A.shape[0]), I)
    else:
        raise Exception("Give only I or only D!")

    if b is None:
        ret_value = (A[I].T[I].T,)
    else:
        if isinstance(b, spmatrix):
            # generalized eigenvalue problem: don't modify rhs
            Aout = A[I].T[I].T
            bout = b[I].T[I].T
        elif isinstance(b, ndarray):
            Aout = A[I].T[I].T
            bout = b[I] - A[I].T[D].T @ x[D]
        else:
            raise Exception("The second arg type not supported.")
        ret_value = (Aout, bout)

    if expand:
        ret_value += (x, I)

    return ret_value if len(ret_value) > 1 else ret_value[0]


# additional utilities


def rcm(A: spmatrix,
        b: ndarray) -> Tuple[spmatrix, ndarray, ndarray]:
    """Reverse Cuthill-McKee ordering."""
    p = spg.reverse_cuthill_mckee(A, symmetric_mode=False)
    return A[p].T[p].T, b[p], p


def adaptive_theta(est, theta=0.5, max=None):
    """For choosing which elements to refine in an adaptive strategy."""
    if max is None:
        return np.nonzero(theta*np.max(est) < est)[0]
    else:
        return np.nonzero(theta*max < est)[0]


def project(fun,
            basis_from: Basis = None,
            basis_to: Basis = None,
            diff: int = None,
            I: ndarray = None) -> ndarray:
    """Projection from one basis to another.

    Parameters
    ----------
    fun
        A solution vector or a function handle.
    basis_from
        The finite element basis to project from.
    basis_to
        The finite element basis to project to.
    diff
        Differentiate with respect to the given dimension.
    I
        Index set for limiting the projection to a subset.

    Returns
    -------
    ndarray
        The projected solution vector.

    """

    @bilinear_form
    def mass(u, du, v, dv, w):
        p = u * v
        return sum(p) if isinstance(basis_to.elem, ElementVectorH1) else p

    @linear_form
    def funv(v, dv, w):
        p = fun(*w.x) * v
        return sum(p) if isinstance(basis_to.elem, ElementVectorH1) else p

    @bilinear_form
    def deriv(u, du, v, dv, w):
        return du[diff] * v

    M = asm(mass, basis_to)

    if not isinstance(fun, ndarray):
        f = asm(funv, basis_to)
    else:
        if diff is not None:
            f = asm(deriv, basis_from, basis_to) @ fun
        else:
            f = asm(mass, basis_from, basis_to) @ fun

    if I is not None:
        return solve(*condense(M, f, I=I, expand=False))

    return solve(M, f)


# for backwards compatibility
L2_projection = lambda a, b, c=None: project(a, basis_to=b, I=c)
derivative = lambda a, b, c, d=0: project(a, basis_from=b, basis_to=c, diff=d)
