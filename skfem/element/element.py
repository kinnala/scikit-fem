from typing import Optional, Tuple, Union,\
    List, NamedTuple

import numpy as np
from numpy import ndarray


class DiscreteField(NamedTuple):
    """A function defined at the global quadrature points."""

    f: Optional[ndarray] = None
    df: Optional[ndarray] = None
    ddf: Optional[ndarray] = None

    def __array__(self):
        return self.f

    def __mul__(self, other):
        if isinstance(other, DiscreteField):
            return self.f * other.f
        return self.f * other

    def split(self):
        """Split all components based on their first dimension."""
        return [DiscreteField(*[f[i] for f in self if f is not None])
                for i in range(self.f.shape[0])]

    def zeros_like(self):
        return DiscreteField(*[0. * field.copy()
                               for field in self
                               if field is not None])

    __rmul__ = __mul__


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

    """
    nodal_dofs: int = 0 
    facet_dofs: int = 0
    interior_dofs: int = 0
    edge_dofs: int = 0
    dim: int = -1
    maxdeg: int = -1
    dofnames: List[str] = []

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

    def _bfun_counts(self, mapping):
        return (self.nodal_dofs * mapping.mesh.t.shape[0],
                self.edge_dofs * mapping.mesh.t2e.shape[0]\
                if hasattr(mapping.mesh, 'edges') else 0,
                self.facet_dofs * mapping.mesh.t2f.shape[0]\
                if hasattr(mapping.mesh, 'facets') else 0,
                self.interior_dofs)

    def __mul__(self, other):

        class ElementComposite(Element):

            def _deduce_bfun(self, mapping, i):
                e1 = self.elems[0]._bfun_counts(mapping)
                e2 = self.elems[1]._bfun_counts(mapping)
                if i < e1[0]:
                    return 0, i
                elif i < e1[0] + e2[0]:
                    return 1, i - e1[0]
                elif i < e1[0] + e2[0] + e1[1]:
                    return 0, i - e2[0]
                elif i < e1[0] + e2[0] + e1[1] + e2[1]:
                    return 1, i - e1[0] - e1[1]
                elif i < e1[0] + e2[0] + e1[1] + e2[1] + e1[2]:
                    return 0, i - e2[0] - e2[1]
                elif i < e1[0] + e2[0] + e1[1] + e2[1] + e1[2] + e2[2]:
                    return 1, i - e1[0] - e1[1] - e1[2]
                elif i < e1[0] + e2[0] + e1[1] + e2[1] + e1[2] + e2[2] + e1[3]:
                    return 0, i - e2[0] - e2[1] - e2[2]
                return 1, i - e1[0] - e1[1] - e1[2] - e1[3]

            def gbasis(self, mapping, X, i, **kwargs):
                n, ind = self._deduce_bfun(mapping, i)
                field = self.elems[n].gbasis(mapping, X, ind, **kwargs)[0]
                if n == 0:
                    Z = self.elems[1].gbasis(mapping, X, 0, **kwargs)[0].zeros_like()
                    return (field, Z)
                elif n == 1:
                    Z = self.elems[0].gbasis(mapping, X, 0, **kwargs)[0].zeros_like()
                    return (Z, field)

        e = ElementComposite()
        e.elems = [self, other]

        e.nodal_dofs = self.nodal_dofs + other.nodal_dofs
        e.edge_dofs = self.edge_dofs + other.edge_dofs
        e.facet_dofs = self.facet_dofs + other.facet_dofs
        e.interior_dofs = self.interior_dofs + other.interior_dofs

        e.maxdeg = self.nodal_dofs + other.maxdeg

        return e
