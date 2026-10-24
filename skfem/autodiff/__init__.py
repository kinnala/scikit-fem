"""Linearization of nonlinear forms using automatic differentiation.

The trial and test functions are represented using second-order
forward-mode dual numbers (see :class:`~skfem.autodiff.dual.Dual`)
which are seeded with respect to the scalar entries of the fields at
the quadrature points, e.g., the value and the gradient components.

An evaluation of the form gives the pointwise coefficients of the
linearized form which are then contracted with the basis functions,
similarly to :class:`~skfem.assembly.BilinearForm`.  NumPy functions
can be applied to the dual numbers directly.

"""
import numpy as np
from numpy import ndarray

from skfem.assembly.form.form import Form, FormExtraParams
from skfem.assembly.form.coo_data import COOData
from skfem.element.block_array import BlockArray
from .dual import Dual, FIELD_ATTRS


class NonlinearForm(Form):
    """A nonlinear form linearized using automatic differentiation.

    The form is either ``form(u, v, w)``, linear in the test function
    ``v``, or ``form(u, w)`` with the keyword argument
    ``hessian=True``, an energy functional whose first variation gives
    the residual and second variation gives the Jacobian.

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
        # example 'slots' for 3D Taylor-Hood:
        # | field | attr  | shape  | offset | size |
        # |-------+-------+--------+--------+------|
        # | u     | value | (3,)   |      0 |    3 |
        # | u     | grad  | (3, 3) |      3 |    9 |
        # | p     | value | ()     |     12 |    1 |
        # | p     | grad  | (3,)   |     13 |    3 |
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

        one = np.ones((1, 1))

        def seed(shape, offset, size):
            # each seed is a single entry: store the nonzeros only
            return BlockArray((K,) + shape + (1, 1),
                              1 + len(shape),
                              {(offset + j,) + ix: one
                               for j, ix in enumerate(np.ndindex(*shape))})

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

        # basis functions sampled at the seeds; most entries are zero, e.g.,
        # a component of ElementVector touches only its own seeds
        samples = []
        for i in range(Nbfun):
            sample = {}
            for k, field in enumerate(slots):
                for n, shape, offset, size in field:
                    f = basis.basis[i][k].get(n)
                    if isinstance(f, BlockArray):
                        f = f.promote(len(shape))
                        for key, b in f.blocks.items():
                            ix = (np.ravel_multi_index(key, shape)
                                  if len(key) > 0 else 0)
                            sample[offset + int(ix)] = b
                    else:
                        f = np.broadcast_to(f, shape + (nt, nq))
                        f = f.reshape(size, nt, nq)
                        for j in range(size):
                            sample[offset + j] = f[j]
            samples.append(sample)

        # group the basis functions by the seeds they touch
        groups = {}
        for i, sample in enumerate(samples):
            groups.setdefault(tuple(sorted(sample)), []).append(i)
        groups = [(np.array(ixs), list(seeds), np.array([
            [np.broadcast_to(samples[i][a], (nt, nq)) for a in seeds]
            for i in ixs
        ])) for seeds, ixs in groups.items()]

        def restrict(d, *seeds):
            """Entries of the derivative ``d`` at the given seeds."""
            if isinstance(d, BlockArray):
                out = np.zeros(tuple(len(s) for s in seeds) + (nt, nq),
                               dtype=d.dtype)
                pos = [{a: j for j, a in enumerate(s)} for s in seeds]
                for key, b in d.blocks.items():
                    try:
                        out[tuple(p[a] for p, a in zip(pos, key))] = b
                    except KeyError:
                        continue
                return out
            d = np.broadcast_to(d, (K,) * len(seeds) + (nt, nq))
            return d[np.ix_(*seeds)]

        dtype = self.dtype
        data = np.zeros((Nbfun, Nbfun, nt), dtype=dtype)
        data1 = np.zeros((Nbfun, nt), dtype=dtype)
        for ixs, seeds, phi in groups:
            phidx = phi * dx
            if res is not None:
                data1[ixs] = np.einsum('beq,ibeq->ie',
                                       restrict(res, seeds),
                                       phidx,
                                       optimize=True)
            if jac is None:
                continue
            for jxs, aseeds, psi in groups:
                # data[j, i] corresponds to trial function j and test
                # function i
                data[np.ix_(jxs, ixs)] = np.einsum(
                    'jaeq,iaeq->jie',
                    psi,
                    np.einsum('abeq,ibeq->iaeq',
                              restrict(jac, aseeds, seeds),
                              phidx,
                              optimize=True),
                    optimize=True
                )

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
