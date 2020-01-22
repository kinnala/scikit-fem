from typing import Optional, Tuple, Union,\
    List, NamedTuple

from numpy import ndarray


class DiscreteField(NamedTuple):
    """A function defined at the global quadrature points."""

    f: Optional[ndarray] = None
    df: Optional[ndarray] = None
    ddf: Optional[ndarray] = None

    def __mul__(self, other):
        if isinstance(other, DiscreteField):
            return self.f * other.f
        return self.f * other

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
