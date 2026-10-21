"""Linearization of nonlinear forms using built-in autodifferentation.

The trial and test functions are represented using second-order
forward-mode dual numbers (see :class:`~skfem.ad.dual.Dual`) which
are seeded with respect to the scalar entries of the fields at the
quadrature points, e.g., the value and the gradient components.

A single evaluation of the form gives the pointwise coefficients of
the linearized form which are then contracted with the basis
functions, similarly to :class:`~skfem.assembly.BilinearForm`.  NumPy
functions can be applied to the dual numbers directly.

"""
import numpy as np
from numpy import ndarray

from skfem.assembly.form.form import Form, FormExtraParams
from skfem.assembly.form.coo_data import COOData
from .dual import Dual, FIELD_ATTRS


class NonlinearForm(Form):
    """A nonlinear form linearized using autodifferentiation.

    The form is either ``form(u, v, w)``, linear in the test function ``v``,
    or ``form(u, w)`` with the keyword argument ``hessian=True``, an energy
    functional whose first variation gives the residual and second variation
    gives the Jacobian.

    """

    def assemble(self, basis, x=None, **kwargs):
        """Assemble the Jacobian and the right-hand side.

        Parameters
        ----------
        basis
            The basis used for all variables.
        x
            Optional point at which the form is linearized, default is zero.

        """
        mat, vec = self._assemble(basis, x=x, **kwargs)
        return (
            COOData._assemble_scipy_csr(*mat),
            COOData(*vec).todefault()
        )

    def elemental(self, basis, x=None, **kwargs):
        """Return COOData for the Jacobian and the right-hand side.

        Parameters
        ----------
        basis
            The basis used for all variables.
        x
            Optional point at which the form is linearized, default is zero.

        """
        mat, vec = self._assemble(basis, x=x, **kwargs)
        return COOData(*mat), COOData(*vec)

    def _assemble(self, basis, x=None, **kwargs):

        if x is None:
            x = basis.zeros()
        if isinstance(x, ndarray):
            x = basis.interpolate(x)
        if not isinstance(x, tuple):
            x = (x,)

        nt = basis.nelems
        dx = basis.dx
        nq = dx.shape[-1]
        Nbfun = basis.Nbfun

        w = FormExtraParams({
            **basis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, basis),
        })

        # the scalar entries of all fields are the seeds; find their layout
        slots = []
        K = 0
        for field in basis.basis[0]:
            slots.append([])
            for n, arr in enumerate(field.astuple):
                if arr is None:
                    continue
                shape = arr.shape[:-2]
                size = int(np.prod(shape))
                slots[-1].append((n, shape, K, size))
                K += size

        def seed(shape, offset, size):
            s = np.zeros((K, size))
            s[offset:offset + size] = np.eye(size)
            return s.reshape((K,) + shape + (1, 1))

        def make_fields(values, group):
            out = []
            for k, field in enumerate(slots):
                parts = {}
                for n, shape, offset, size in field:
                    if values is None:
                        val = np.zeros(shape + (1, 1))
                    else:
                        val = np.asarray(values[k].get(n))
                    s = seed(shape, offset, size)
                    if group == 'a':
                        parts[n] = Dual(val, ga=s)
                    elif group == 'b':
                        parts[n] = Dual(val, gb=s)
                    else:
                        parts[n] = Dual(val, ga=s, gb=s)
                main = parts.pop(0, None)
                if main is None:
                    raise NotImplementedError("Fields without values are "
                                              "not supported.")
                for n in parts:
                    setattr(main, FIELD_ATTRS[n - 1], parts[n])
                out.append(main)
            return out

        if self.params.get('hessian', False):
            # energy functional: both seed groups are the trial function
            out = self.form(*make_fields(x, 'ab'), w)
            res = out.ga if isinstance(out, Dual) else None
        else:
            # the form is linear in the test function, evaluate at zero
            out = self.form(*make_fields(x, 'a'),
                            *make_fields(None, 'b'),
                            w)
            res = out.gb if isinstance(out, Dual) else None
        jac = out.hab if isinstance(out, Dual) else None

        # basis functions sampled at the seeds: (Nbfun, K, nt, nq)
        phi = np.zeros((Nbfun, K, nt, nq))
        for i in range(Nbfun):
            for k, field in enumerate(slots):
                for n, shape, offset, size in field:
                    phi[i, offset:offset + size] = np.broadcast_to(
                        basis.basis[i][k].get(n),
                        shape + (nt, nq)
                    ).reshape(size, nt, nq)
        phidx = phi * dx

        dtype = self.dtype
        if jac is None:
            data = np.zeros((Nbfun, Nbfun, nt), dtype=dtype)
        else:
            jac = np.broadcast_to(jac, (K, K, nt, nq))
            # data[j, i] corresponds to trial function j and test function i
            data = np.einsum('jaeq,iaeq->jie',
                             phi,
                             np.einsum('abeq,ibeq->iaeq', jac, phidx),
                             optimize=True).astype(dtype, copy=False)
        if res is None:
            data1 = np.zeros((Nbfun, nt), dtype=dtype)
        else:
            res = np.broadcast_to(res, (K, nt, nq))
            data1 = np.einsum('beq,ibeq->ie',
                              res,
                              phidx,
                              optimize=True).astype(dtype, copy=False)

        dofs = basis.element_dofs
        rows = np.broadcast_to(dofs[None, :, :], (Nbfun, Nbfun, nt))
        cols = np.broadcast_to(dofs[:, None, :], (Nbfun, Nbfun, nt))

        return (
            (
                np.array([rows.flatten(), cols.flatten()]),
                data.flatten('C'),
                (basis.N, basis.N),
                (basis.Nbfun, basis.Nbfun),
            ),
            (
                np.array([dofs.flatten()]),
                -data1.flatten('C'),
                (basis.N,),
                (basis.Nbfun,)
            )
        )
