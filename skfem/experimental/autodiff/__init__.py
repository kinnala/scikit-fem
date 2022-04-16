from skfem.assembly.form.form import Form, FormExtraParams
from skfem.assembly.form.coo_data import COOData
from numpy import ndarray
import numpy as np

from autograd import make_jvp


class NonlinearForm(Form):

    def assemble(self, u0, basis, **kwargs):
        """Assemble the Jacobian and the right-hand side.

        Parameters
        ----------
        u0
            The point at which the form is linearized.
        basis
            The basis used for all variables.

        """
        # interpolate and cast to tuple
        # make u0 compatible with u in forms
        if isinstance(u0, ndarray):
            u0 = basis.interpolate(u0)
            if isinstance(u0, tuple):
                u0 = tuple(c.astuple for c in u0)
            else:
                u0 = (u0.astuple,)

        nt = basis.nelems
        dx = basis.dx
        w = FormExtraParams({
            **basis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, basis),
        })

        # initialize COO data structures
        sz = basis.Nbfun * basis.Nbfun * nt
        data = np.zeros((basis.Nbfun, basis.Nbfun, nt), dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int64)
        cols = np.zeros(sz, dtype=np.int64)

        sz1 = basis.Nbfun * nt
        data1 = np.zeros(sz1, dtype=self.dtype)
        rows1 = np.zeros(sz1, dtype=np.int64)

        def _make_jacobian(V):
            if 'hessian' in self.params:
                F = make_jvp(lambda U: self.form(*U, w))
                return make_jvp(lambda W: F(W)(V)[1])(u0)
            return make_jvp(lambda U: self.form(*U, *V, w))(u0)

        # # JAX version
        # def _make_jacobian(V):
        #     if 'hessian' in self.params:
        #         return linearize(
        #             lambda W: jvp(lambda U: self.form(*U, w), (W,), (V,))[1],
        #             u0
        #         )
        #     return linearize(lambda U: self.form(*U, *V, w), u0)

        # loop over the indices of local stiffness matrix
        for i in range(basis.Nbfun):
            DF = _make_jacobian(tuple(c.astuple for c in basis.basis[i]))
            for j in range(basis.Nbfun):
                y, DFU = DF(tuple(c.astuple for c in basis.basis[j]))
                # Jacobian
                ixs = slice(nt * (basis.Nbfun * j + i),
                            nt * (basis.Nbfun * j + i + 1))
                rows[ixs] = basis.element_dofs[i]
                cols[ixs] = basis.element_dofs[j]
                data[j, i, :] = np.sum(DFU * dx, axis=1)
            # rhs
            ixs1 = slice(nt * i, nt * (i + 1))
            rows1[ixs1] = basis.element_dofs[i]
            data1[ixs1] = np.sum(y * dx, axis=1)

        data = data.flatten('C')

        return (
            COOData._assemble_scipy_csr(
                np.array([rows, cols]),
                data,
                (basis.N, basis.N),
                (basis.Nbfun, basis.Nbfun),
            ),
            COOData(
                np.array([rows1]),
                data1,
                (basis.N,),
                (basis.Nbfun,)
            ).todefault()
        )
