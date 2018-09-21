import numpy as np
from scipy.sparse import coo_matrix
from inspect import signature

from typing import NamedTuple, Optional, Dict, List, Tuple, Union

from numpy import ndarray

from scipy.sparse import csr_matrix
from .global_basis import GlobalBasis

def asm(kernel,
        ubasis: GlobalBasis,
        vbasis: Optional[GlobalBasis] = None,
        w: Optional[ndarray] = None,
        dw: Optional[ndarray] = None,
        ddw: Optional[ndarray] = None,
        nthreads: Optional[int] = 1,
        assemble: Optional[bool] = True) -> csr_matrix:
    """Assemble finite element matrices.

    Parameters
    ----------
    kernel
        The bilinear/linear form.
    ubasis
        GlobalBasis object for 'u'.
    vbasis
        GlobalBasis object for 'v'.
    w
        Accessible in form definition as w.w.
    dw
        Accessible in form definition as w.dw.
    ddw
        Accessible in form definition as w.ddw.
    nthreads
        Number of threads to use in assembly. Due to Python global interpreter
        lock (GIL), this is only useful if kernel is numba function compiled
        with nogil = True, see Examples.

    Examples
    --------

    Creating multithreadable kernel function.

    >>> from numba import njit
    >>> @njit(nogil=True)
        def assemble(A, ix, u, du, v, dv, w, dx):
            for k in range(ix.shape[0]):
                i, j = ix[k]
                A[i, j] = np.sum((du[j][0]*dv[i][0] + du[j][1]*dv[i][1] + du[j][2]*dv[i][2]) * dx, axis=1)
    >>> assemble.bilinear = True

    """
    import threading
    from itertools import product

    if vbasis is None:
        vbasis = ubasis

    nt = ubasis.nelems
    dx = ubasis.dx

    class FormParameters(NamedTuple):
        x: ndarray
        h: ndarray
        n: Optional[ndarray] = None
        w: Optional[ndarray] = None
        dw: Optional[ndarray] = None
        ddw: Optional[ndarray] = None

    w = FormParameters(w=w, dw=dw, ddw=ddw, **ubasis.default_parameters())

    if kernel.bilinear:
        # initialize COO data structures
        data = np.zeros((vbasis.Nbfun, ubasis.Nbfun, nt))
        rows = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)
        cols = np.zeros(ubasis.Nbfun * vbasis.Nbfun * nt)

        # create sparse matrix indexing
        for j in range(ubasis.Nbfun):
            for i in range(vbasis.Nbfun):
                # find correct location in data,rows,cols
                ixs = slice(nt * (vbasis.Nbfun * j + i), nt * (vbasis.Nbfun * j + i + 1))
                rows[ixs] = vbasis.element_dofs[i, :]
                cols[ixs] = ubasis.element_dofs[j, :]

        # create indices for linear loop over local stiffness matrix
        ixs = [i for j, i in product(range(ubasis.Nbfun), range(vbasis.Nbfun))]
        jxs = [j for j, i in product(range(ubasis.Nbfun), range(vbasis.Nbfun))]
        indices = np.array([ixs, jxs]).T

        # split local stiffness matrix elements to threads
        threads = [threading.Thread(target=kernel, args=(data, ij,
                                                         *ubasis.basis,
                                                         *vbasis.basis,
                                                         w, dx)) for ij
                   in np.array_split(indices, nthreads, axis=0)]

        # start threads and wait for finishing
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        if assemble:
            K = coo_matrix((np.transpose(data, (1, 0, 2)).flatten('C'), (rows, cols)),
                              shape=(vbasis.N, ubasis.N))
            K.eliminate_zeros()
            return K.tocsr()
        else:
            return (np.transpose(data, (1, 0, 2)).flatten('C'), (rows, cols))
    else:
        data = np.zeros((vbasis.Nbfun, nt))
        rows = np.zeros(vbasis.Nbfun * nt)
        cols = np.zeros(vbasis.Nbfun * nt)

        for i in range(vbasis.Nbfun):
            # find correct location in data,rows,cols
            ixs = slice(nt * i, nt * (i + 1))
            rows[ixs] = vbasis.element_dofs[i, :]
            cols[ixs] = np.zeros(nt)

        indices = range(vbasis.Nbfun)

        threads = [threading.Thread(target=kernel, args=(data, ix, *vbasis.basis, w, dx)) for ix
                   in np.array_split(indices, nthreads, axis=0)]

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        return coo_matrix((data.flatten('C'), (rows, cols)),
                          shape=(vbasis.N, 1)).toarray().T[0]


def bilinear_form(form):
    """Bilinear form decorator.
    
    This decorator is used for defining bilinear forms that can be assembled
    using :func:`~skfem.assembly.asm`.

    """
    nargs = len(signature(form).parameters)
    if nargs == 5:
        def kernel(A, ix, u, du, v, dv, w, dx):
            for k in range(ix.shape[0]):
                i, j = ix[k]
                A[i, j] = np.sum(form(u[j], du[j], v[i], dv[i], w) * dx, axis=1)
    elif nargs == 7:
        def kernel(A, ix, u, du, ddu, v, dv, ddv, w, dx):
            for k in range(ix.shape[0]):
                i, j = ix[k]
                A[i, j] = np.sum(form(u[j], du[j], ddu[j], v[i], dv[i], ddv[i], w) * dx, axis=1)
    else:
        raise NotImplementedError("Given number of form arguments not supported.")
    kernel.bilinear = True
    return kernel


def linear_form(form):
    """Linear form decorator.
    
    This decorator is used for defining linear forms that can be assembled
    using :func:`~skfem.assembly.asm`.

    """
    nargs = len(signature(form).parameters)
    if nargs == 3:
        def kernel(b, ix, v, dv, w, dx):
            for i in ix:
                b[i] = np.sum(form(v[i], dv[i], w) * dx, axis=1)
    elif nargs == 4:
        def kernel(b, ix, v, dv, ddv, w, dx):
            for i in ix:
                b[i] = np.sum(form(v[i], dv[i], ddv[i], w) * dx, axis=1)
    else:
        raise NotImplementedError("Given number of form arguments not supported.")
    kernel.bilinear = False
    return kernel
