from typing import Optional, List, Type

import numpy as np
from numpy import ndarray

from skfem.mesh import Mesh
from .discrete_field import DiscreteField


class Element():
    """Finite element.

    Attributes
    ----------
    nodal_dofs
        Number of DOFs per node.
    facet_dofs
        Number of DOFs per facet.
    interior_dofs
        Number of DOFs inside the element.
    edge_dofs
        Number of DOFs per edge.
    dim
        The spatial dimension.
    maxdeg
        Polynomial degree of the basis. Used to calculate quadrature rules.
    dofnames
        A list of strings that indicate DOF types. Different possibilities:
        - 'u' indicates that it is the point value
        - 'u^1' indicates the first vectorial component
        - 'u^n' indicates the normal component
        - 'u^t' indicates the tangential component
        - 'u_x' indicates the derivative wrt x
        - 'u_n' indicates the normal derivative
        - ...
    mesh_type
        Mesh type for calculating number of edges, etc.

    """
    nodal_dofs: int = 0
    facet_dofs: int = 0
    interior_dofs: int = 0
    edge_dofs: int = 0
    dim: int = -1
    maxdeg: int = -1
    dofnames: List[str] = []
    mesh_type: Type = Mesh

    def orient(self, mapping, i, tind=None):
        """Orient basis functions. By default all = 1."""
        if tind is None:
            return 1 + 0 * mapping.mesh.t[0]
        else:
            return 1 + 0 * tind

    def gbasis(self,
               mapping,
               X: ndarray,
               i: int,
               tind: Optional[ndarray] = None) -> DiscreteField:
        """Evaluate the global basis functions, given local points X.

        The global points - at which the global basis is evaluated at - are
        defined through x = F(X), where F corresponds to the given mapping.

        Parameters
        ----------
        mapping
            Local-to-global mapping, an object of type
            :class:`~skfem.mapping.Mapping`.
        X
            An array of local points. The following shapes are supported: (Ndim
            x Npoints) and (Ndim x Nelems x Npoints), i.e. local points shared
            by all elements or different local points in each element.
        i
            Only the i'th basis function is evaluated.
        tind
            Optionally, choose a subset of elements at which to evaluate the
            basis.

        Returns
        -------
        DiscreteField
            The global basis function evaluted at the quadrature points.

        """
        raise NotImplementedError("Element must implement gbasis.")

    @classmethod
    def _index_error(cls):
        raise ValueError("Index larger than the number of basis functions.")

    def _bfun_counts(self) -> ndarray:
        """Count number of nodal/edge/facet/interior basis functions."""
        return np.array([self.nodal_dofs * self.mesh_type.t.shape[0],
                         self.edge_dofs * self.mesh_type.t2e.shape[0]
                         if hasattr(self.mesh_type, 'edges') else 0,
                         self.facet_dofs * self.mesh_type.t2f.shape[0]
                         if hasattr(self.mesh_type, 'facets') else 0,
                         self.interior_dofs])

    def __mul__(self, other):

        from .element_composite import ElementComposite

        a = self.elems if isinstance(self, ElementComposite) else [self]
        b = other.elems if isinstance(other, ElementComposite) else [other]

        return ElementComposite(*a, *b)

    @property
    def refdom(self):
        return self.mesh_type.refdom

    @property
    def brefdom(self):
        return self.mesh_type.brefdom
