import numpy as np
from typing import Optional, Tuple, Union, List
from numpy import ndarray


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
    order
        A tuple representing the tensorial order of u, du, (and ddu).
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
    # 0 - scalar, 1 - vector, 2 - tensor, etc
    order: Union[Tuple[int, int], Tuple[int, int, int]] = (-1, -1)
    dofnames: List[str] = []

    def orient(self, mapping, i, tind=None):
        """Orient basis functions. By default all = 1."""
        if tind is None:
            return 1 + 0*mapping.mesh.t[0, :]
        else:
            return 1 + 0*tind

    def gbasis(self,
               mapping,
               X: ndarray,
               i: int,
               tind: Optional[ndarray] = None) -> Union[Tuple[ndarray, ndarray],
                                                        Tuple[ndarray, ndarray, ndarray]]:
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
        (u, du) or (u, du, ddu)
            The number of return arguments depends on the length of self.order.
            The shape of k'th return argument depends on the value of
            self.order[k].

        """
        raise NotImplementedError("Element must implement gbasis.")
