# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
import scipy.sparse.csgraph as spg


def bilinear_form(form):
    """Bilinear form decorator"""
    def kernel(A, ix, u, du, v, dv, w, dx):
        for k in range(ix.shape[0]):
            i, j = ix[k]
            A[i, j] = np.sum(form(u[j], du[j], v[i], dv[i], w) * dx, axis=1)
    kernel.bilinear = True
    return kernel


def linear_form(form):
    """Linear form decorator"""
    def kernel(b, ix, v, dv, w, dx):
        for i in ix:
            b[i] = np.sum(form(v[i], dv[i], w) * dx, axis=1)
    kernel.bilinear = False
    return kernel


def nonlinear_form(nonlin):
    """
    Create tangent system using automatic differentiation.

    The new form is bilinear and has the parameters (u, du, v, dv, w).

    It is expected that w[0] contains u_0, w[1] contains du_0/dx, etc.

    Note: Requires autograd. Use autograd.numpy instead of numpy for any special operations.
    """
    from autograd import elementwise_grad as egrad

    @bilinear_form
    def bilin(u, du, v, dv, w):
        order = (len(u.shape)-2, len(du.shape)-2)
        if order[0] > 0:
            dim = u.shape[0]
        elif order[1] > 0:
            dim = du.shape[0]
        else:
            raise Exception("Could not deduce the dimension!")
        if order[0] == 0 and order[1] == 1:
            # scalar H1
            first_arg = egrad(nonlin, argnum=0)(w[0], w[1:(dim+1)], v, dv, w[(dim+1):])*u
            second_arg = np.sum(egrad(nonlin, argnum=1)(w[0], w[1:(dim+1)], v, dv, w[(dim+1):])*du, axis=0)
        elif order[0] == 1 and order[1] == 0:
            # Hdiv / Hcurl
            first_arg = np.sum(egrad(nonlin, argnum=0)(w[0:dim], w[dim], v, dv, w[(dim+1):])*u, axis=0)
            second_arg = egrad(nonlin, argnum=1)(w[0:dim], w[dim], v, dv, w[(dim+1):])*du
        else:
            raise Exception("The linearization of the given order not supported.")
        # derivative chain rule
        return first_arg + second_arg

    @linear_form
    def lin(v, dv, w):
        order = (len(v.shape)-2, len(dv.shape)-2)
        if order[0] > 0:
            dim = v.shape[0]
        elif order[1] > 0:
            dim = dv.shape[0]
        else:
            raise Exception("Could not deduce the dimension!")
        if order[0] == 0 and order[1] == 1:
            # scalar H1
            return nonlin(w[0], w[1:(dim+1)], v, dv, w[(dim+1):])
        elif order[0] == 1 and order[1] == 0:
            # Hdiv / Hcurl
            return nonlin(w[0:dim], w[dim], v, dv, w[(dim+1):])
        else:
            raise Exception("The linearization of the given order not supported.")

    bilin.rhs = lin

    return bilin


def smoothplot(x, basis, nref=3, m=None):
    M, X = basis.refinterp(x, nref)
    if m is not None:
        ax = m.draw()
        M.plot(X, smooth=True, edgecolors='', ax=ax)
    else:
        M.plot(X, smooth=True, edgecolors='')


def rcm_reordering(A, symmetric=False):
    """Compute reverse Cuthill-McKee reordering using SciPy."""
    return spg.reverse_cuthill_mckee(A, symmetric_mode=symmetric)


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
    p = rcm_reordering(A)
    return A[p].T[p].T, b[p], p


def solver_direct_scipy():
    def solver(A, b):
        return spl.spsolve(A, b)
    return solver


def solver_direct_cholmod():
    def solver(A, b):
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        return factor(b)
    return solver


def solve(A, b, solver=None):
    """
    Example for using Umfpack:

    .. code-block:: python
    def solver_umfpack(A, b):
        return spl.spsolve(A, b, use_umfpack=True)

    Example for using CHOLMOD through scikit-sparse:

    .. code-block:: python
    def solver_cholmod(A, b):
        from sksparse.cholmod import cholesky
        factor = cholesky(A)
        return factor(b)

    """
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
