from typing import Optional, List, Type, Tuple
from types import MethodType
from copy import deepcopy

import numpy as np
from numpy import ndarray

from ..refdom import Refdom
from .discrete_field import DiscreteField


class Element:
    """Evaluate finite element basis.

    This class should not be initialized directly.  Use different subclasses,
    e.g.,

    - :class:`ElementLineP1`
    - :class:`ElementTriP1`
    - :class:`ElementQuad1`
    - :class:`ElementTetP1`
    - :class:`ElementHex1`

    Attributes
    ----------
    nodal_dofs
        Number of DOFs per node.  Used to define the global DOF numbering.
    facet_dofs
        Number of DOFs per facet.  Used to define the global DOF numbering.
    interior_dofs
        Number of DOFs inside the element.  Used to define the global DOF
        numbering.
    edge_dofs
        Number of DOFs per edge.  Used to define the global DOF numbering.
    dim
        The spatial dimension.
    maxdeg
        Polynomial degree of the basis.  Used to automatically find quadrature
        rules.
    dofnames
        A list of strings indicating the DOF types. See :ref:`finddofs`.

    """
    nodal_dofs: int = 0
    facet_dofs: int = 0
    interior_dofs: int = 0
    edge_dofs: int = 0
    maxdeg: int = -1
    dofnames: List[str] = []
    refdom: Type[Refdom] = Refdom
    doflocs: ndarray

    @property
    def dim(self):
        return self.refdom.dim()

    def orient(self, mapping, i, tind=None):
        """Orient basis functions. By default all = 1."""
        if tind is None:
            return 1 + 0 * mapping.mesh.t[0]
        else:
            return 1 + 0 * tind

    def __call__(self):
        return self

    def gbasis(self,
               mapping,
               X: ndarray,
               i: int,
               tind: Optional[ndarray] = None) -> Tuple[DiscreteField, ...]:
        """Evaluate the global basis functions, given local points ``X``.

        The global points corresponding to ``X`` are obtained through the
        local-to-global mapping given by the parameter ``mapping``.

        Parameters
        ----------
        mapping
            Local-to-global mapping, an object of type
            :class:`~skfem.mapping.Mapping`.
        X
            An array of local points. The following shapes are supported:
            ``(Ndim, Npoints)`` and ``(Ndim, Nelems, Npoints)``, i.e. local
            points shared by all elements or different local points in each
            element.
        i
            Only the ``i`` th basis function is evaluated.
        tind
            Optionally, choose a subset of elements at which to evaluate the
            basis.

        Returns
        -------
        Tuple[DiscreteField, ...]]
            The global basis function evaluted at the quadrature points.

        """
        raise NotImplementedError("Element must implement gbasis.")

    @classmethod
    def _index_error(cls):
        raise ValueError("Index larger than the number of basis functions.")

    def condensed(self):
        """Return two elements: one for interior and one for other DOFs."""

        eo = deepcopy(self)
        eo.interior_dofs = 0

        if hasattr(eo, 'elems'):
            for i in range(len(eo.elems)):
                eo.elems[i].interior_dofs = 0

        ei = deepcopy(self)
        ei.nodal_dofs = 0
        ei.facet_dofs = 0
        ei.edge_dofs = 0

        if hasattr(ei, 'elems'):
            for i in range(len(ei.elems)):
                ei.elems[i].nodal_dofs = 0
                ei.elems[i].facet_dofs = 0
                ei.elems[i].edge_dofs = 0

        def gbasis(obj,
                   mapping,
                   X: ndarray,
                   i: int,
                   tind: Optional[ndarray] = None):
            return self.gbasis(mapping,
                               X,
                               i + self._bfun_counts()[:3].sum(),
                               tind)

        ei.gbasis = MethodType(gbasis, ei)

        return ei, eo

    def _bfun_counts(self) -> ndarray:
        """Count number of nodal/edge/facet/interior basis functions."""
        return np.array([self.nodal_dofs * self.refdom.nnodes,
                         self.edge_dofs * self.refdom.nedges,
                         self.facet_dofs * self.refdom.nfacets,
                         self.interior_dofs])

    def __mul__(self, other):

        from .element_composite import ElementComposite

        a = self.elems if isinstance(self, ElementComposite) else [self]
        b = other.elems if isinstance(other, ElementComposite) else [other]

        return ElementComposite(*a, *b)
