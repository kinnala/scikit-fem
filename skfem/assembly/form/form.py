import logging
import numbers
from copy import deepcopy
from functools import partial
from typing import Any, Callable, Optional, Union, Type
from inspect import signature
from warnings import warn

import numpy as np
from numpy import ndarray

from ...element import DiscreteField
from .coo_data import COOData


logger = logging.getLogger(__name__)


class FormExtraParams(dict):
    """Passed to forms as 'w'."""

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError("Attribute '{}' not found in 'w'.".format(attr))


class Form:

    form: Optional[Callable] = None

    def __init__(self,
                 form: Optional[Union[Callable, 'Form']] = None,
                 dtype: Union[Type[np.float64],
                              Type[np.complex64]] = np.float64,
                 nthreads: int = 0,
                 **params):
        self.form = form.form if isinstance(form, Form) else form
        self.nargs = (len(signature(self.form).parameters)
                      if self.form is not None
                      else None)
        self.dtype = dtype
        self.nthreads = nthreads
        self.params = params

    def partial(self, *args, **kwargs):
        form = deepcopy(self)
        name = form.form.__name__
        form.form = partial(form.form, *args, **kwargs)
        form.form.__name__ = name
        return form

    def block(self, *args):
        warn("Form.block is deprecated. "
             "Use instead Form.assemble_blocks.", DeprecationWarning)
        form = deepcopy(self)
        name = form.form.__name__
        form.form = lambda *arg: self.form(
            *[arg[k] if args[k] == j else arg[k].zeros()
              for k in range(len(arg) - 1)
              for j in range(int((self.nargs - 1) / (len(arg) - 1)))],
            arg[-1]
        )
        form.form.__name__ = name
        return form

    def __call__(self, *args):
        if self.form is None:  # decorate
            return type(self)(form=args[0],
                              dtype=self.dtype,
                              nthreads=self.nthreads,
                              **self.params)
        return self.assemble(self.kernel(*args))

    def assemble(self, *args, **kwargs) -> Any:
        """Assemble system matrix as SciPy CSR sparse matrix.

        If the basis is defined on a distributed mesh, see
        :func:`skfem.distributed.distribute`, returns a PETSc matrix,
        PETSc vector or a scalar summed over all processes.

        """
        assert self.form is not None
        logger.info("Assembling '{}'.".format(self.form.__name__))
        data = COOData(*self._assemble(*args, **kwargs))  # type: ignore
        dofs = self._distributed_dofs(*args)
        out = data.todefault() if dofs is None else data._todistributed(dofs)
        logger.info("Assembling finished.")
        return out

    @staticmethod
    def _distributed_dofs(*bases):
        """Dofs of the bases if they are defined on a distributed mesh."""
        dofs = [basis.dofs for basis in bases
                if getattr(getattr(basis, 'dofs', None), '_comm', None)
                is not None]
        if len(dofs) == 0:
            return None
        if any(not np.array_equal(d._globnums, dofs[0]._globnums)
               for d in dofs):
            raise NotImplementedError("Distributed assembly supports only "
                                      "forms with a single basis.")
        return dofs[0]

    @staticmethod
    def _components(basis):
        """Global DOFs of each solution component of the basis."""
        out = basis.split_indices()
        if sum(len(ix) for ix in out) != basis.N or (
                len(np.unique(np.concatenate(out))) != basis.N):
            raise ValueError("The solution components do not partition "
                             "the DOFs.")
        return out

    @staticmethod
    def _local_components(basis):
        """Local basis functions of each solution component of the basis,
        and the numbering of the global DOFs within the component."""
        comp = np.empty(basis.N, dtype=np.int64)
        renum = np.empty(basis.N, dtype=np.int64)
        components = Form._components(basis)
        for c, ix in enumerate(components):
            comp[ix] = c
            renum[ix] = np.arange(len(ix))
        edofs = basis.element_dofs
        lcomp = comp[edofs]
        if edofs.shape[1] > 0 and not (lcomp == lcomp[:, :1]).all():
            raise ValueError("The solution component of a local basis "
                             "function varies between elements.")
        lcomp = lcomp[:, 0] if edofs.shape[1] > 0 else np.zeros(0)
        return [(np.nonzero(lcomp == c)[0], renum, len(ix))
                for c, ix in enumerate(components)]

    def elemental(self, *args, **kwargs) -> COOData:
        """Build elemental matrices without full assembly."""
        return COOData(*self._assemble(*args, **kwargs))  # type: ignore

    def coo_data(self, *args, **kwargs):
        """Build elemental matrices without full assembly.

        .. deprecated:: 13.0.0
           Use :meth:`~skfem.assembly.form.Form.elemental` instead.

        """
        warn("Form.coo_data is deprecated. Use Form.elemental instead.",
             DeprecationWarning, stacklevel=2)
        return self._coo_data(*args, **kwargs)

    def _coo_data(self, *args, **kwargs) -> COOData:
        # used by asm; unlike Form.elemental, never overridden in subclasses
        return COOData(*self._assemble(*args, **kwargs))

    @staticmethod
    def _integrate(f, dx: ndarray, axis: int = 1) -> ndarray:
        """Sum ``f * dx`` over the quadrature points of each element."""
        if (isinstance(f, ndarray)
                and f.shape == dx.shape
                and f.dtype != object):
            # avoids allocating the temporary array f * dx
            return np.einsum('ij,ij->i', f, dx)
        return np.sum(f * dx, axis=axis)

    @staticmethod
    def _normalize_asm_kwargs(w, basis):
        """Support additional input formats for 'w'."""
        for k in w:
            if isinstance(w[k], DiscreteField):
                if w[k].shape[-1] != basis.X.shape[-1]:
                    raise ValueError("Quadrature mismatch: '{}' should have "
                                     "same number of integration points as "
                                     "the basis object.".format(k))
            elif isinstance(w[k], numbers.Number):
                # scalar parameter is passed to form
                continue
            elif isinstance(w[k], tuple):
                # asm() product index is of type tuple
                continue
            elif isinstance(w[k], ndarray) and len(w[k].shape) == 1:
                # interpolate DOF arrays at quadrature points
                w[k] = basis.interpolate(w[k])
            elif isinstance(w[k], ndarray) and len(w[k].shape) > 1:
                w[k] = DiscreteField(w[k])
            else:
                raise ValueError("The given type '{}' for the list of extra "
                                 "form parameters w cannot be converted to "
                                 "DiscreteField.".format(type(w[k])))
        return w
