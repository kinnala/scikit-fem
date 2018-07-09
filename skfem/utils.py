# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.sparse.csgraph as spg
import warnings


def condense(A, b=None, x=None, I=None, D=None):
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
        return A[I].T[I].T, b[I] - A[I].T[D].T.dot(x[D])


def rcm(A, b):
    p = spg.reverse_cuthill_mckee(A, symmetric_mode=False)
    return A[p].T[p].T, b[p], p


def solver_direct_scipy():
    def solver(A, b):
        return spl.spsolve(A, b)
    return solver


def solver_direct_umfpack():
    def solver(A, b):
        return spl.spsolve(A, b, use_umfpack=True)
    return solver


def solver_direct_cholmod():
    """Cholmod-based direct solver for symmetric systems."""
    def solver(A, b):
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        return factor(b)
    return solver


def build_ilu_pc(A):
    """Incomplete LU preconditioner."""
    P = spl.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=20)
    P_x = lambda x: P.solve(x)
    M = spl.LinearOperator((A.shape[0], A.shape[0]), matvec=P_x)
    return M


def build_diag_pc(A):
    """Diagonal preconditioner."""
    return sp.spdiags(1.0/A.diagonal(), 0, A.shape[0], A.shape[0])


def solver_iter_pcg(pc=None, guess=None, maxiter=100, tol=1e-8, verbose=False):
    """Conjugate gradient solver.
    
    Parameters
    ----------
    pc : (optional) sparse matrix, LinearOperator
        A preconditioner for the conjugate gradient algorithm.
        By default, a diagonal preconditioner is built using
        skfem.utils.build_diag_pc. User can supply a fixed
        preconditioner using this parameter.
    guess : (optional) numpy array
        An initial guess. By default, zero is used as an initial
        guess.
    maxiter : (optional, default=100) int
    tol : (optional, default=1e-8) float
    verbose : (optional, default=False) bool
    """
    def callback(x):
        if verbose:
            print(np.linalg.norm(x))

    if pc is None:
        def solver(A, b):
            sol, info = spl.cg(A, b, x0=guess, maxiter=maxiter, M=build_diag_pc(A), tol=tol, callback=callback)
            if info > 0:
                warnings.warn("Convergence not achieved (TOL=" + str(tol) + ", MAXITER=" + str(maxiter) + ")")
            return sol
    else:
        def solver(A, b):
            sol, info = spl.cg(A, b, x0=guess, maxiter=maxiter, M=pc, tol=tol, callback=callback)
            if info > 0:
                warnings.warn("Convergence not achieved (TOL=" + str(tol) + ", MAXITER=" + str(maxiter) + ")")
            return sol

    return solver


def solve(A, b, solver=None):
    if solver is None:
        solver = solver_direct_scipy()

    return solver(A, b)


def adaptive_theta(est, theta=0.5, max=None):
    if max is None:
        return np.nonzero(theta*np.max(est) < est)[0]
    else:
        return np.nonzero(theta*max < est)[0]


def initialize(basis, *bcs):
    """Initialize a solution vector with boundary conditions in place."""
    y = np.zeros(basis.dofnum.N)
    boundary_ix = np.array([])

    for bc in bcs:
        x, D = bc(basis)
        y += x
        boundary_ix = np.concatenate((boundary_ix, D))

    return y, basis.dofnum.complement_dofs(boundary_ix)


def derivative(x, basis1, basis0, i=0):
    """Calculate the i'th partial derivative by projecting from basis1 to basis0."""
    from skfem.assembly import asm

    @bilinear_form
    def deriv(u, du, v, dv, w):
        return du[i]*v

    @bilinear_form
    def mass(u, du, v, dv, w):
        return u*v

    A = asm(deriv, basis1, basis0)
    M = asm(mass, basis0)

    return solve(M, A @ x)
     
