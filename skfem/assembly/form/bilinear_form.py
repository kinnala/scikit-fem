from typing import Any, Callable, Optional

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix, csr_matrix

from .form import Form
from .form_parameters import FormParameters
from ..global_basis import GlobalBasis


class BilinearForm(Form):

    def kernel(self,
               A: ndarray,
               ix: ndarray,
               ubasis: GlobalBasis,
               vbasis: GlobalBasis,
               w: FormParameters,
               dx: ndarray) -> None:
        for k in range(ix.shape[0]):
            i, j = ix[k]
            A[i, j] = np.sum(self.form(*ubasis[j], *vbasis[i], w) * dx,
                             axis=1)

    def assemble(self,
                 ubasis: GlobalBasis,
                 vbasis: Optional[GlobalBasis] = None,
                 w: Optional[Any] = (None, None, None),
                 nthreads: Optional[int] = 1) -> csr_matrix:
        """return sparse CSR matrix discretizing form

        :param w: A tuple of ndarrays. In the form definition:

          * :code:`w[0]` is accessible as :code:`w.w`,

          * :code:`w[1]` is accessible as :code:`w.dw`, and

          * :code:`w[2]` is accessible as :code:`w.ddw`.

        The output of :meth:`~skfem.assembly.GlobalBasis.interpolate`
        can be passed directly to this parameter.

        """

        import threading
        from itertools import product

        if vbasis is None:
            vbasis = ubasis

        nt = ubasis.nelems
        dx = ubasis.dx
        w = self.parameters(w, ubasis)

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
        threads = [threading.Thread(
            target=self.kernel,
            args=(data, ij, ubasis.basis, vbasis.basis, w, dx))
                   for ij in np.array_split(indices, nthreads, axis=0)]

        # start threads and wait for finishing
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        K = coo_matrix((np.transpose(data, (1, 0, 2)).flatten('C'),
                        (rows, cols)),
                       shape=(vbasis.N, ubasis.N))
        K.eliminate_zeros()
        return K.tocsr()


def bilinear_form(form: Callable) -> BilinearForm:
    return BilinearForm(form)
