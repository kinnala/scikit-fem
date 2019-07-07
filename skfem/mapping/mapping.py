from typing import Optional
from numpy import ndarray


class Mapping():
    def F(self,
          X: ndarray,
          tind: Optional[ndarray] = None) -> ndarray:
        """Perform mapping from the reference element
        to global elements.

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
        raise NotImplementedError("!")

    def invF(self,
             x: ndarray,
             tind: Optional[ndarray] = None) -> ndarray:
        """Perform an inverse mapping from global elements to
        reference element.

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
        raise NotImplementedError("!")

    def G(self,
          X: ndarray,
          find: Optional[ndarray] = None) -> ndarray:
        """Perform a mapping from the reference facet to global facet.

        Parameters
        ----------
        X
            Local points on the reference element (Ndim x Nqp).
        find
            A set of facet indices to map to

        Returns
        -------
        ndarray
            Global points (Ndim x Nelems x Nqp).

        """
        raise NotImplementedError("!")

    def detDG(self,
              X: ndarray,
              find: Optional[ndarray] = None) -> ndarray:
        raise NotImplementedError("!")
    
    def normals(self, X, tind, find, t2f):
        raise NotImplementedError("!")

    def detDF(self, X, tind=None):
        raise NotImplementedError("!")

    def DF(self, X, tind=None):
        raise NotImplementedError("!")

    def invDF(self, X, tind=None):
        raise NotImplementedError("!")
