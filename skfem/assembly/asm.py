import numpy as np
from scipy.sparse import coo_matrix
from inspect import signature

from typing import NamedTuple, Optional, Dict, List, Tuple, Union, Any

from numpy import ndarray

from scipy.sparse import csr_matrix
from .global_basis import GlobalBasis


def asm(kernel,
        ubasis: GlobalBasis,
        vbasis: Optional[GlobalBasis] = None,
        w: Optional[Any] = (None, None, None),
        nthreads: Optional[int] = 1) -> csr_matrix:
    """Assemble finite element matrices and vectors.

    Parameters
    ----------
    kernel
        The bilinear/linear form.
    ubasis
        GlobalBasis object for 'u'.
    vbasis
        GlobalBasis object for 'v'.
    w
        A tuple of ndarrays. In the form definition, w[0] is accessible as
        w.w, w[1] is accessible as w.dw, and w[2] is accessible as w.ddw.
        The output of :meth:`~skfem.assembly.GlobalBasis.interpolate` can be
        passed directly to this parameter.
    nthreads
        Number of threads to use in assembly. Due to Python global interpreter
        lock (GIL), this is only useful if kernel is numba function compiled
        with nogil = True, see Examples.

    Examples
    --------
    Creating a multithreadable kernel function.

    >>> from numba import njit
    >>> @njit(nogil=True)
        def form(A, ix, u, v, w, dx):
            for k in range(ix.shape[0]):
                i, j = ix[k]
                A[i, j] = np.sum((u[j][1][0]*v[i][1][0] +\
                                  u[j][1][1]*v[i][1][1] +\
                                  u[j][1][2]*v[i][1][2]) * dx, axis=1)
    >>> form.bilinear = True

    """
    import threading
    from itertools import product

    if vbasis is None:
        vbasis = ubasis

    nt = ubasis.nelems
    dx = ubasis.dx
    nargs = len(signature(kernel).parameters)

    if type(w) is list:
        w = zip(*w)
    elif type(w) is ndarray:
        w = (w, None, None)

    class FormParameters(NamedTuple):
        w: Optional[ndarray] = None
        dw: Optional[ndarray] = None
        ddw: Optional[ndarray] = None
        h: Optional[ndarray] = None
        n: Optional[ndarray] = None
        x: Optional[ndarray] = None
    
    w = FormParameters(*w, **ubasis.default_parameters())
    
    if nargs == 6:
        # initialize COO data structures
        data = np.zeros((vbasis.Nbfun, ubasis.Nbfun, nt))
        rows = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)
        cols = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)

        # create sparse matrix indexing
        for j in range(ubasis.Nbfun):
            for i in range(vbasis.Nbfun):
                # find correct location in data,rows,cols
                ixs = slice(nt * (vbasis.Nbfun * j + i),
                            nt * (vbasis.Nbfun * j + i + 1))
                rows[ixs] = vbasis.element_dofs[i, :]
                cols[ixs] = ubasis.element_dofs[j, :]

        # create indices for linear loop over local stiffness matrix
        ixs = [i for j, i in product(range(ubasis.Nbfun), range(vbasis.Nbfun))]
        jxs = [j for j, i in product(range(ubasis.Nbfun), range(vbasis.Nbfun))]
        indices = np.array([ixs, jxs]).T

        # split local stiffness matrix elements to threads
        threads = [threading.Thread(target=kernel, args=(data, ij, ubasis.basis, vbasis.basis, w, dx))
                   for ij in np.array_split(indices, nthreads, axis=0)]

        # start threads and wait for finishing
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        K = coo_matrix((np.transpose(data, (1, 0, 2)).flatten('C'), (rows, cols)),
                        shape=(vbasis.N, ubasis.N))
        K.eliminate_zeros()
        return K.tocsr()

    elif nargs == 5:
        data = np.zeros((vbasis.Nbfun, nt))
        rows = np.zeros(vbasis.Nbfun * nt)
        cols = np.zeros(vbasis.Nbfun * nt)

        for i in range(vbasis.Nbfun):
            # find correct location in data,rows,cols
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.element_dofs[i, :]
            cols[ixs] = np.zeros(nt)

        indices = range(vbasis.Nbfun)

        threads = [threading.Thread(target=kernel, args=(data, ix, vbasis.basis, w, dx))
                   for ix in np.array_split(indices, nthreads, axis=0)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return coo_matrix((data.flatten('C'), (rows, cols)),
                           shape=(vbasis.N, 1)).toarray().T[0]

    else:
        return kernel(w, dx)


def bilinear_form(form):
    """Bilinear form decorator.
    
    This decorator is used for defining bilinear forms that can be assembled
    using :func:`~skfem.assembly.asm`.

    """
    def kernel(A, ix, ubasis, vbasis, w, dx):
        for k in range(ix.shape[0]):
            i, j = ix[k]
            A[i, j] = np.sum(form(*ubasis[j], *vbasis[i], w) * dx, axis=1)
    return kernel


def linear_form(form):
    """Linear form decorator.
    
    This decorator is used for defining linear forms that can be assembled
    using :func:`~skfem.assembly.asm`.

    """
    def kernel(b, ix, vbasis, w, dx):
        for i in ix:
            b[i] = np.sum(form(*vbasis[i], w) * dx, axis=1)
    return kernel


def functional(form):
    """Functional decorator.

    This decorator is used for defining functionals that can be evaluated
    using :func:`~skfem.assembly.asm`.

    """
    def kernel(w, dx):
        return np.sum(form(w) * dx, axis=1)
    return kernel
    
