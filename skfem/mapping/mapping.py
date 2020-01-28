from typing import Optional

from numpy import ndarray


class Mapping():
    """Base class for mappings between reference and global elements."""

    def F(self,
          X: ndarray,
          tind: Optional[ndarray] = None) -> ndarray:
        """Perform mapping from the reference element to global elements.

        Parameters
        ----------
        X
            Local points on the reference element (Ndim x Nqp).
        tind
            A set of element indices to map to

        Returns
        -------
        ndarray
            Global points (Ndim x Nelems x Nqp)

        """
        raise NotImplementedError

    def invF(self,
             x: ndarray,
             tind: Optional[ndarray] = None) -> ndarray:
        """Perform an inverse mapping from global elements to reference element.

        Parameters
        ----------
        x
            The global points (Ndim x Nelems x Nqp).
        tind
            A set of element indices to map from

        Returns
        -------
        ndarray
            The corresponding local points (Ndim x Nelems x Nqp).

        """
        raise NotImplementedError

    def G(self,
          X: ndarray,
          find: Optional[ndarray] = None) -> ndarray:
        """Perform a mapping from the reference facet to global facet.

        Parameters
        ----------
        X
            Local points on the reference element (Ndim x Nqp).
        find
            A set of facet indices to map to.

        Returns
        -------
        ndarray
            Global points (Ndim x Nelems x Nqp).

        """
        raise NotImplementedError

    def detDG(self,
              X: ndarray,
              find: Optional[ndarray] = None) -> ndarray:
        """The jacobian determinant of G.

        Parameters
        ----------
        X
            Local points on the reference element (Ndim x Nqp).
        find
            A set of facet indices to restrict to.

        Returns
        -------
        ndarray
            Jacobian determinants at global points (Ndim x Nelems x Nqp).

        """
        raise NotImplementedError

    def normals(self,
                X: ndarray,
                tind: ndarray,
                find: ndarray,
                t2f: ndarray) -> ndarray:
        """Calculate normal vectors on element boundaries."""
        raise NotImplementedError

    def DF(self,
           X: ndarray,
           tind: Optional[ndarray] = None):
        """The jacobian of F."""
        raise NotImplementedError

    def invDF(self,
              X: ndarray,
              tind: Optional[ndarray] = None):
        """The inverse of the jacobian of F."""
        raise NotImplementedError

    def detDF(self,
              X: ndarray,
              tind: Optional[ndarray] = None):
        """The determinant of the jacobian of F."""
        raise NotImplementedError
