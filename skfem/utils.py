# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.sparse.csgraph as spg


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
    def solver(A, b):
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        return factor(b)
    return solver


def solve(A, b, solver=None):
    if solver is None:
        solver = solver_direct_scipy()

    return solver(A, b)


def build_ilu_pc(A):
    """Build a preconditioner for the matrix using incomplete LU decomposition.
    The output can be fed directly to e.g. skfem.utils.cg."""
    P = spl.spilu(A.tocsc(), drop_tol=1e-4, fill_factor=20)
    P_x = lambda x: P.solve(x)
    M = spl.LinearOperator((A.shape[0], A.shape[0]), matvec=P_x)
    return M


def cg(A, b, tol, maxiter, x0=None, D=None, I=None, pc=None, verbose=True, viewiters=False):
    """Conjugate gradient solver wrapped for FEM purposes."""
    def callback(x):
        if viewiters:
            print("- Vector-2 norm: " + str(np.linalg.norm(x)))

    if D is not None:
        I = np.setdiff1d(np.arange(A.shape[0]), D)
    else:
        if I is None:
            I = np.arange(A.shape[0])

    if pc is None:
        # diagonal preconditioner
        M = sp.spdiags(1/(A[I].T[I].diagonal()), 0, I.shape[0], I.shape[0])
    else:
        M = pc

    if I is None:
        u = spl.cg(A, b, x0=x0, maxiter=maxiter, M=M, tol=tol, callback=callback)
    else:
        if x0 is None:
            u = spl.cg(A[I].T[I].T, b[I], maxiter=maxiter, M=M, tol=tol,
                       callback=callback)
        else:
            u = spl.cg(A[I].T[I].T, b[I], x0=x0[I], maxiter=maxiter, M=M,
                       tol=tol, callback=callback)

    if verbose:
        if u[1] > 0:
            print("* WARNING! Maximum number of iterations "\
                  + str(maxiter) + " reached.")

    if I is None: 
        return u[0]
    else:
        U = np.zeros(A.shape[0])
        U[I] = u[0]
        return U


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
     
