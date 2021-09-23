"""This module contains utility functions such as convenient access to
SciPy linear solvers."""

import warnings
from typing import Optional, Union, Tuple, Callable, Dict

import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as spg
import scipy.sparse.linalg as spl
from numpy import ndarray
from scipy.sparse import spmatrix

from skfem.assembly import asm, BilinearForm, LinearForm, DofsView
from skfem.assembly.basis import AbstractBasis
from skfem.element import ElementVector


# custom types for describing input and output values


Solution = Union[ndarray, Tuple[ndarray, ndarray]]
LinearSolver = Callable[..., ndarray]
EigenSolver = Callable[..., Tuple[ndarray, ndarray]]
LinearSystem = Union[spmatrix,
                     Tuple[spmatrix, ndarray],
                     Tuple[spmatrix, spmatrix]]
CondensedSystem = Union[LinearSystem,
                        Tuple[spmatrix, ndarray, ndarray],
                        Tuple[spmatrix, ndarray, ndarray, ndarray],
                        Tuple[spmatrix, spmatrix, ndarray, ndarray]]
DofsCollection = Union[ndarray, DofsView, Dict[str, DofsView]]


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
    }
    params.update(kwargs)

    def solver(K, M, **solve_time_kwargs):
        params.update(solve_time_kwargs)
        from scipy.sparse.linalg import eigs
        return eigs(K, M=M, **params)

    return solver


def solver_eigen_scipy_sym(**kwargs) -> EigenSolver:
    """Solve symmetric generalized eigenproblem using SciPy (ARPACK).

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
    """The default linear solver of SciPy."""

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

def solve_eigen(A: spmatrix,
                M: spmatrix,
                x: Optional[ndarray] = None,
                I: Optional[ndarray] = None,
                solver: Optional[Union[EigenSolver]] = None,
                **kwargs) -> Tuple[ndarray, ndarray]:

    if solver is None:
        solver = solver_eigen_scipy(**kwargs)

    if x is not None and I is not None:
        L, X = solver(A, M, **kwargs)
        y = np.tile(x.copy()[:, None], (1, X.shape[1]))
        y[I] = X
        return L, y
    return solver(A, M, **kwargs)


def solve_linear(A: spmatrix,
                 b: ndarray,
                 x: Optional[ndarray] = None,
                 I: Optional[ndarray] = None,
                 solver: Optional[Union[LinearSolver]] = None,
                 **kwargs) -> ndarray:

    if solver is None:
        solver = solver_direct_scipy(**kwargs)

    if x is not None and I is not None:
        y = x.copy()
        y[I] = solver(A, b, **kwargs)
        return y
    return solver(A, b, **kwargs)


def solve(A: spmatrix,
          b: Union[ndarray, spmatrix],
          x: Optional[ndarray] = None,
          I: Optional[ndarray] = None,
          solver: Optional[Union[LinearSolver, EigenSolver]] = None,
          **kwargs) -> Solution:
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
    if isinstance(b, spmatrix):
        return solve_eigen(A, b, x, I, solver, **kwargs)  # type: ignore
    elif isinstance(b, ndarray):
        return solve_linear(A, b, x, I, solver, **kwargs)  # type: ignore
    raise NotImplementedError("Provided argument types not supported")


def _flatten_dofs(S: Optional[DofsCollection]) -> Optional[ndarray]:
    if S is None:
        return None
    if isinstance(S, ndarray):
        return S
    elif isinstance(S, DofsView):
        return S.flatten()
    elif isinstance(S, dict):
        def _flatten_helper(S, key):
            if key in S and isinstance(S[key], DofsView):
                return S[key].flatten()
            raise NotImplementedError
        return np.unique(
            np.concatenate([_flatten_helper(S, key) for key in S])
        )
    raise NotImplementedError("Unable to flatten the given set of DOFs.")


def _init_bc(A: spmatrix,
             b: Optional[Union[ndarray, spmatrix]] = None,
             x: Optional[ndarray] = None,
             I: Optional[DofsCollection] = None,
             D: Optional[DofsCollection] = None) -> Tuple[Optional[ndarray],
                                                          ndarray,
                                                          ndarray,
                                                          ndarray]:

    D = _flatten_dofs(D)
    I = _flatten_dofs(I)

    if I is None and D is None:
        raise Exception("Either I or D must be given!")
    elif I is None and D is not None:
        I = np.setdiff1d(np.arange(A.shape[0]), D)
    elif D is None and I is not None:
        D = np.setdiff1d(np.arange(A.shape[0]), I)
    else:
        raise Exception("Give only I or only D!")

    assert isinstance(I, ndarray)
    assert isinstance(D, ndarray)

    if x is None:
        x = np.zeros(A.shape[0])
    elif b is None:
        b = np.zeros_like(x)

    return b, x, I, D


def enforce(A: spmatrix,
            b: Optional[Union[ndarray, spmatrix]] = None,
            x: Optional[ndarray] = None,
            I: Optional[DofsCollection] = None,
            D: Optional[DofsCollection] = None,
            diag: float = 1.,
            overwrite: bool = False) -> LinearSystem:
    r"""Enforce degrees-of-freedom of a linear system.

    An alternative to :func:`~skfem.utils.condense` which sets the matrix
    diagonals to one and right-hand side vector to the enforced
    degree-of-freedom value.

    .. note::

        The original system is both returned
        (for compatibility with :func:`skfem.utils.solve`) and optionally (if
        `overwrite`) modified (for performance).

    Parameters
    ----------
    A
        The system matrix
    b
        Optionally, the right hand side vector.
    x
        The values of the enforced degrees-of-freedom. If not given, assumed
        to be zero.
    I
        Specify either this or ``D``: The set of degree-of-freedom indices to
        solve for.
    D
        Specify either this or ``I``: The set of degree-of-freedom indices to
        enforce (rows/diagonal set to zero/one).
    overwrite
        Optionally, the original system is both modified (for performance) and
        returned (for compatibility with :func:`skfem.utils.solve`).  By
        default, ``False``.

    Returns
    -------
    LinearSystem
        A linear system with the enforced rows/diagonals set to zero/one.

    """
    b, x, I, D = _init_bc(A, b, x, I, D)

    Aout = A if overwrite else A.copy()

    # set rows on lhs to zero
    start = Aout.indptr[D]
    stop = Aout.indptr[D + 1]
    count = stop - start
    idx = np.ones(count.sum(), dtype=np.int64)
    idx[np.cumsum(count)[:-1]] -= count[:-1]
    idx = np.repeat(start, count) + np.cumsum(idx) - 1
    Aout.data[idx] = 0.

    # set diagonal value
    d = Aout.diagonal()
    d[D] = diag
    Aout.setdiag(d)

    if b is not None:
        if isinstance(b, spmatrix):
            # mass matrix (eigen- or initial value problem)
            bout = enforce(b, D=D, diag=0., overwrite=overwrite)
        else:
            # set rhs to the given value
            bout = b if overwrite else b.copy()
            bout[D] = x[D]
        return Aout, bout

    return Aout


def penalize(A: spmatrix,
             b: Optional[Union[ndarray, spmatrix]] = None,
             x: Optional[ndarray] = None,
             I: Optional[DofsCollection] = None,
             D: Optional[DofsCollection] = None,
             epsilon: Optional[float] = None,
             overwrite: bool = False) -> LinearSystem:
    r"""Penalize degrees-of-freedom of a linear system.

    Parameters
    ----------
    A
        The system matrix
    b
        Optionally, the right hand side vector.
    x
        The values of the penalized degrees-of-freedom. If not given, assumed
        to be zero.
    I
        Specify either this or ``D``: The set of degree-of-freedom indices to
        solve for.
    D
        Specify either this or ``I``: The set of degree-of-freedom indices to
        enforce (rows/diagonal set to zero/one).
    epsilon
        Very small value, the reciprocal of which penalizes deviations from
        the Dirichlet condition
    overwrite
        Optionally, the original system is both modified (for performance) and
        returned (for compatibility with :func:`skfem.utils.solve`).  By
        default, ``False``.

    Returns
    -------
    LinearSystem
        A linear system with the penalized diagonal and RHS entries set to
        very large values, 1/epsilon and x/epsilon, respectively.

    """
    b, x, I, D = _init_bc(A, b, x, I, D)

    Aout = A if overwrite else A.copy()

    d = Aout.diagonal()
    if epsilon is None:
        epsilon = 1e-10 / np.linalg.norm(d[D], np.inf)
    d[D] = 1/epsilon
    Aout.setdiag(d)

    if b is None:
        return Aout

    bout = b if overwrite else b.copy()
    # Nothing needs doing for mass matrix, but RHS vector needs penalty factor
    if not isinstance(b, spmatrix):
        bout[D] = x[D] / epsilon
    return Aout, bout


def condense(A: spmatrix,
             b: Optional[Union[ndarray, spmatrix]] = None,
             x: Optional[ndarray] = None,
             I: Optional[DofsCollection] = None,
             D: Optional[DofsCollection] = None,
             expand: bool = True) -> CondensedSystem:
    r"""Eliminate degrees-of-freedom from a linear system.

    The user should provide the linear system ``A`` and ``b``
    and either the set of DOFs to eliminate (``D``) or the set
    of DOFs to keep (``I``).  Optionally, nonzero values for
    the eliminated DOFs can be supplied via ``x``.

    .. note::

        Supports also generalized eigenvalue problems
        where ``b`` is a matrix.

    Example
    -------

    Suppose that the solution vector :math:`x` can be
    split as

    .. math::

       x = \begin{bmatrix}
           x_I\\
           x_D
       \end{bmatrix}

    where :math:`x_D` are known and :math:`x_I` are unknown.  This allows
    splitting the linear system as

    .. math::

       \begin{bmatrix}
           A_{II} & A_{ID}\\
           A_{DI} & A_{DD}
       \end{bmatrix}
       \begin{bmatrix}
           x_I\\
           x_D
       \end{bmatrix}
       =
       \begin{bmatrix}
           b_I\\
           b_D
       \end{bmatrix}

    which leads to the condensed system

    .. math::

       A_{II} x_I = b_I - A_{ID} x_D.


    As an example, let us assemble the matrix :math:`A` and the vector
    :math:`b` corresponding to the Poisson equation :math:`-\Delta u = 1`.

    .. doctest::

       >>> import skfem as fem
       >>> from skfem.models.poisson import laplace, unit_load
       >>> m = fem.MeshTri().refined(2)
       >>> basis = fem.CellBasis(m, fem.ElementTriP1())
       >>> A = laplace.assemble(basis)
       >>> b = unit_load.assemble(basis)

    The condensed system is obtained with :func:`skfem.utils.condense`.  Below
    we provide the DOFs to eliminate via the keyword argument
    ``D``.

    .. doctest::

       >>> AII, bI, xI, I = fem.condense(A, b, D=m.boundary_nodes())
       >>> AII.toarray()
       array([[ 4.,  0.,  0.,  0., -1., -1., -1., -1.,  0.],
              [ 0.,  4.,  0.,  0., -1.,  0., -1.,  0.,  0.],
              [ 0.,  0.,  4.,  0.,  0., -1.,  0., -1.,  0.],
              [ 0.,  0.,  0.,  4., -1., -1.,  0.,  0.,  0.],
              [-1., -1.,  0., -1.,  4.,  0.,  0.,  0.,  0.],
              [-1.,  0., -1., -1.,  0.,  4.,  0.,  0.,  0.],
              [-1., -1.,  0.,  0.,  0.,  0.,  4.,  0., -1.],
              [-1.,  0., -1.,  0.,  0.,  0.,  0.,  4., -1.],
              [ 0.,  0.,  0.,  0.,  0.,  0., -1., -1.,  4.]])
        >>> bI
        array([0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625, 0.0625,
               0.0625])

    By default, the eliminated DOFs are set to zero.
    Different values can be provided through the keyword argument ``x``;
    see :ref:`ex14`.

    Parameters
    ----------
    A
        The system matrix
    b
        The right hand side vector, or zero if x is given, or the mass matrix
        for generalized eigenvalue problems.
    x
        The values of the condensed degrees-of-freedom. If not given, assumed
        to be zero.
    I
        The set of degree-of-freedom indices to include.
    D
        The set of degree-of-freedom indices to dismiss.
    expand
        If ``True`` (default), returns also `x` and `I`. As a consequence,
        :func:`skfem.utils.solve` will expand the solution vector
        automatically.

    Returns
    -------
    CondensedSystem
        The condensed linear system and (optionally) information about
        the boundary values.

    """
    b, x, I, D = _init_bc(A, b, x, I, D)

    ret_value: CondensedSystem = (None,)

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
            raise Exception("Type of second arg not supported.")
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
        return np.nonzero(theta * np.max(est) < est)[0]
    else:
        return np.nonzero(theta * max < est)[0]


def projection(fun,
               basis_to: Optional[AbstractBasis] = None,
               basis_from: Optional[AbstractBasis] = None,
               diff: Optional[int] = None,
               I: Optional[ndarray] = None,
               expand: bool = False) -> ndarray:
    """Perform projections onto a finite element basis.

    Parameters
    ----------
    fun
        A solution vector or a function handle.
    basis_to
        The finite element basis to project to.
    basis_from
        The finite element basis to project from.
    diff
        Differentiate with respect to the given dimension.
    I
        Index set for limiting the projection to a subset.
    expand
        Passed to :func:`skfem.utils.condense`.

    Returns
    -------
    ndarray
        The projected solution vector.

    """

    @BilinearForm
    def mass(u, v, w):
        from skfem.helpers import dot, ddot
        p = 0
        if len(u.value.shape) == 2:
            p = u * v
        elif len(u.value.shape) == 3:
            p = dot(u, v)
        elif len(u.value.shape) == 4:
            p = ddot(u, v)
        return p

    if isinstance(fun, LinearForm):
        funv = fun
    else:
        @LinearForm
        def funv(v, w):
            p = fun(w.x) * v
            return sum(p) if isinstance(basis_to.elem, ElementVector) else p

    @BilinearForm
    def deriv(u, v, w):
        from skfem.helpers import grad
        du = grad(u)
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
        return solve_linear(*condense(M, f, I=I, expand=expand))

    return solve_linear(M, f)


def project(fun,
            basis_from: Optional[AbstractBasis] = None,
            basis_to: Optional[AbstractBasis] = None,
            diff: Optional[int] = None,
            I: Optional[ndarray] = None,
            expand: bool = False) -> ndarray:
    warnings.warn("project is deprecated in favor of projection.",
                  DeprecationWarning)
    return projection(
        fun,
        basis_to=basis_to,
        basis_from=basis_from,
        diff=diff,
        I=I,
        expand=expand,
    )
