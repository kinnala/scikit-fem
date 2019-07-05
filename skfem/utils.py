"""This module contains utility functions such as convenient access to
SciPy linear solvers."""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.sparse.csgraph as spg
import warnings
from skfem.assembly import asm, bilinear_form, linear_form
from skfem.element import ElementVectorH1

from typing import Optional, Union, Tuple, Callable
from numpy import ndarray
from scipy.sparse import spmatrix
from skfem.assembly.global_basis import GlobalBasis

LinearSolver = Callable[[spmatrix, ndarray], ndarray]
EigenSolver = Callable[[spmatrix, spmatrix], Tuple[ndarray, ndarray]]


def condense(A: spmatrix,
             b: Optional[Union[ndarray, spmatrix]] = None,
             x: Optional[ndarray] = None,
             I: Optional[ndarray] = None,
             D: Optional[ndarray] = None) -> Union[spmatrix,
                                                   Tuple[spmatrix, ndarray],
                                                   Tuple[spmatrix, spmatrix]]:
    """Eliminate DOF's from a linear system.

    Supports also generalized eigenvalue problems.
    
    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector or the mass matrix
        for generalized eigenvalue problems.
    I
        The set of DOF numbers to keep
    D
        The set of DOF numbers to dismiss
        
    Returns
    -------
    spmatrix or (spmatrix, ndarray) or (spmatrix, spmatrix)
        The condensed system.
        
    """
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
        return A[I].T[I].T
    else:
        if isinstance(b, spmatrix):
            # generalized eigenvalue problem: don't modify rhs
            return A[I].T[I].T, b[I].T[I].T 
        elif isinstance(b, ndarray):
            return A[I].T[I].T, b[I] - A[I].T[D].T @ x[D]
        else:
            raise Exception("The second arg type not supported.")


def rcm(A: spmatrix,
        b: ndarray) -> Tuple[spmatrix, ndarray, ndarray]:
    p = spg.reverse_cuthill_mckee(A, symmetric_mode=False)
    return A[p].T[p].T, b[p], p


def solver_eigen_scipy(sigma: float,
                       n: Optional[int] = 3,
                       mode: Optional[str] = 'normal') -> EigenSolver:
    """Solve generalized eigenproblem using SciPy (ARPACK).
    
    Parameters
    ----------
    sigma
        The parameter for spectral shift, choose a value near the
        expected eigenvalues.
    n
        The number of eigenpairs to solve.

    Returns
    -------
    EigenSolver
        A solver function that can be passed to :func:`solve`.
    
    """
    def solver(K, M):
        from scipy.sparse.linalg import eigsh
        return eigsh(K, k=n, M=M, sigma=sigma, mode=mode)
    return solver


def solver_direct_scipy() -> LinearSolver:
    def solver(A, b):
        return spl.spsolve(A, b)
    return solver


def solver_direct_umfpack() -> LinearSolver:
    """SciPy interface to umfpack."""
    def solver(A, b):
        return spl.spsolve(A, b, use_umfpack=True)
    return solver


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

    Any remaining keyword arguments are passed on to the solver, in
    particular x0, the starting guess, and M, the preconditioner.  If
    the latter is omitted, a diagonal preconditioner is supplied using
    :func:`skfem.utils.build_pc_diag`.


    Returns
    -------
    LinearSolver
        A solver function that can be passed to :func:`solve`.

    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))
    
    def solver(A, b):
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
    return solver_iter_krylov(spl.cg, **kwargs)


def solve(A: spmatrix,
          b: Union[ndarray, spmatrix],
          solver: Optional[Union[LinearSolver, EigenSolver]] = None) -> ndarray:
    """Solve a linear system or a generalized eigenvalue problem.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector or the mass matrix of a generalized
        eigenvalue problem.
    solver
        Choose one of the following solvers:

            - :func:`skfem.utils.solver_direct_scipy` (default)
            - :func:`skfem.utils.solver_eigen_scipy` (default)
            - :func:`skfem.utils.solver_direct_umfpack`
            - :func:`skfem.utils.solver_iter_pcg`

    """
    if solver is None:
        if isinstance(b, spmatrix):
            solver = solver_eigen_scipy(10.0)
        elif isinstance(b, ndarray):
            solver = solver_direct_scipy()

    return solver(A, b)


def adaptive_theta(est, theta=0.5, max=None):
    if max is None:
        return np.nonzero(theta*np.max(est) < est)[0]
    else:
        return np.nonzero(theta*max < est)[0]


def derivative(x: ndarray,
               basis1: GlobalBasis,
               basis0: GlobalBasis,
               i: Optional[int] = 0) -> ndarray:
    """Calculate the i'th partial derivative through projection.

    Parameters
    ----------
    x
        The solution vector.
    basis1
        The basis corresponding to the solution x (e.g. P_1).
    basis0
        The basis corresponding to the derivative field (e.g. P_0).
    i
        Return i'th partial derivative.

    Returns
    -------
    ndarray
        A new solution vector corresponding to the derivative.

    """

    @bilinear_form
    def deriv(u, du, v, dv, w):
        return du[i]*v

    @bilinear_form
    def mass(u, du, v, dv, w):
        return u*v

    A = asm(deriv, basis1, basis0)
    M = asm(mass, basis0)

    return solve(M, A @ x)
     

def L2_projection(fun,
                  basis: GlobalBasis,
                  ix: Optional[ndarray] = None) -> ndarray:
    """Initialize a solution vector with L2 projection.

    Parameters
    ----------
    fun
        The function to project.
    basis
        The finite element basis
    ix
        Do the projection only on a subset of DOF's.

    Returns
    -------
    ndarray
        The projected solution vector.

    """

    if ix is None:
        ix = np.arange(basis.N)
    
    @bilinear_form
    def mass(u, du, v, dv, w):
        p = u * v
        return sum(p) if isinstance(basis.elem, ElementVectorH1) else p

    @linear_form
    def funv(v, dv, w):
        p = fun(*w.x) * v
        return sum(p) if isinstance(basis.elem, ElementVectorH1) else p
    
    M = asm(mass, basis)
    f = asm(funv, basis)

    return solve(*condense(M, f, I=ix))
