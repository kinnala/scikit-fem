# -*- coding: utf-8 -*-
"""
Utility functions.
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from copy import deepcopy

def stack(block):
    """Stack scipy sparse matrices.
    
    Parameters
    ----------
    block : tuple of tuples
        A 2-dim tuple array with the scipy matrices to stack.
    """
    return sp.vstack(list(map(sp.hstack, block)))

def zerosparse(mn):
    """Return a zero sparse matrix.
    
    Parameters
    ----------
    mn : tuple of dims
    """
    return sp.coo_matrix(mn)

def cell_shape(x, *rest):
    """Find out the shape of a cell array."""
    if isinstance(x, dict):
        s = len(x)
        return cell_shape(x[0], s, *rest)
    else:
        return rest[::-1]

def const_cell(nparr, *arg):
    """
    Initialize a cell array (i.e. python dictionary)
    with the given parameter array/float by performing
    a deep copy.

    *Example*. Initializing a cell array with zeroes.

    .. code-block:: python

        >>> from fem.utils import const_cell
        >>> const_cell(0.0, 3, 2)
        {0: {0: 0.0, 1: 0.0}, 1: {0: 0.0, 1: 0.0}, 2: {0: 0.0, 1: 0.0}}
    """
    if len(arg) == 1:
        u = {i: deepcopy(nparr) for (i, _) in enumerate(range(arg[0]))}
    elif len(arg) == 0:
        return nparr
    else:
        u = {i: const_cell(nparr, *arg[1:]) for (i, _) in enumerate(range(arg[0]))}
    return u

def direct(A, b, x=None, I=None, D=None, solve=None):
    """Solve system Ax=b with essential boundary conditions.
    
    Parameters
    ----------
    A : scipy sparse matrix
        The system matrix.
    b : numpy array
        The right hand side.
    x : (OPTIONAL) numpy array
        For implementing inhomogeneous essential conditions. Must be of size
        b.shape[0].
    D : (OPTIONAL) numpy array
        The boundary nodes.
    I : (OPTIONAL) numpy array
        The interior nodes. A list of integers to x corresponding to interior
        nodes.
    solve : (OPTIONAL) lambda
        Function taking two parameters (A,b) and solving Ax = b.
        Default functionality uses SciPy direct solver.

    Examples
    --------
    Default 'solve' uses SuperLU from SciPy. You
    can supply another solver using 'solve'-parameter.

    Example for using Umfpack:

    .. code-block:: python
        def solve_umfpack(A, b):
            return spl.spsolve(A, b, use_umfpack=True)

    Example for using CHOLMOD through scikit-sparse:

    .. code-block:: python
        def solve_cholmod(A, b):
            from sksparse.cholmod import cholesky
            factor = cholesky(A)
            return factor(b)
    """
    def default_solve(A, b):
        return spl.spsolve(A, b)

    if solve is None:
        solve = default_solve

    if x is None:
        x = np.zeros(A.shape[0])

    if I is None:
        if D is None:
            x = solve(A, b)
        else:
            I = np.setdiff1d(np.arange(A.shape[0]), D)
            x[I] = solve(A[I].T[I].T, b[I] - A[I].T[D].T.dot(x[D]))
    else:
        D = np.setdiff1d(np.arange(A.shape[0]), I)
        x[I] = solve(A[I].T[I].T, b[I] - A[I].T[D].T.dot(x[D]))

    return x

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
        if u[1] == 0:
            print("* Achieved tolerance " + str(tol) + ".")
        elif u[1] > 0:
            print("* WARNING! Maximum number of iterations "\
                  + str(maxiter) + " reached.")

    if I is None: 
        return u[0]
    else:
        U = np.zeros(A.shape[0])
        U[I] = u[0]
        return U
