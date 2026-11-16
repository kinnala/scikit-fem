from typing import Optional, Tuple

import numpy as np
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
        """Perform an inverse mapping from global elements to reference
        element.

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

    def edge_reference(self,
                       X: ndarray,
                       eind: Optional[ndarray] = None
                       ) -> Tuple[ndarray, ndarray]:
        """Map points on the reference edge to the reference element.

        Each edge is associated with one of the elements sharing it.  The
        reference edge :math:`[0, 1]` is parametrized from the vertex
        ``mesh.edges[0]`` to the vertex ``mesh.edges[1]``.

        Parameters
        ----------
        X
            Local points on the reference edge (1 x Nqp).
        eind
            A set of edge indices.

        Returns
        -------
        ndarray
            The corresponding local points in the reference element
            (Ndim x Nedges x Nqp).
        ndarray
            The indices of the elements, one for each edge.

        """
        A, B, tind = self._edge_endpoints(eind)
        s = np.asarray(X, dtype=np.float64).reshape(-1)
        return (A[:, :, None] + (B - A)[:, :, None] * s[None, None, :],
                tind)

    def _edge_endpoints(self, eind=None):
        """Endpoints of the edges in the reference element coordinates."""
        mesh = self.mesh  # type: ignore
        refdom = mesh.refdom
        if refdom.edges is None or mesh.dim() != 3:
            raise NotImplementedError("Edge mapping is supported only for "
                                      "three-dimensional elements.")
        if not hasattr(self, '_edge_parents'):
            t2e = mesh.t2e
            tind = np.zeros(mesh.nedges, dtype=np.int32)
            loc = np.zeros(mesh.nedges, dtype=np.int32)
            for i in range(t2e.shape[0]):
                tind[t2e[i]] = np.arange(t2e.shape[1], dtype=np.int32)
                loc[t2e[i]] = i
            self._edge_parents = (tind, loc)
        tind, loc = self._edge_parents
        if eind is None:
            eind = np.arange(mesh.nedges, dtype=np.int32)
        eind = np.asarray(eind)
        tind, loc = tind[eind], loc[eind]
        ledges = np.array(refdom.edges)[loc]  # (Nedges, 2)
        # flip the local edges which are reversed w.r.t. mesh.edges
        flip = mesh.t[ledges[:, 0], tind] != mesh.edges[0, eind]
        ledges[flip] = ledges[flip, ::-1]
        P = np.asarray(refdom.p, dtype=np.float64)
        return P[:, ledges[:, 0]], P[:, ledges[:, 1]], tind

    def DH(self,
           X: ndarray,
           eind: Optional[ndarray] = None) -> ndarray:
        """The derivative of H, i.e. the (non-unit) tangent of the edges.

        Parameters
        ----------
        X
            Local points on the reference edge (1 x Nqp).
        eind
            A set of edge indices.

        Returns
        -------
        ndarray
            Tangent vectors (Ndim x Nedges x Nqp).

        """
        A, B, _ = self._edge_endpoints(eind)
        Y, tind = self.edge_reference(X, eind)
        DF = self.DF(Y, tind=tind)  # (Ndim, Ndim, Nedges, Nqp)
        return np.einsum('ijkl,jk->ikl', DF, B - A)

    def H(self,
          X: ndarray,
          eind: Optional[ndarray] = None) -> ndarray:
        """Perform a mapping from the reference edge to global edges.

        Parameters
        ----------
        X
            Local points on the reference edge (1 x Nqp).
        eind
            A set of edge indices to map to.

        Returns
        -------
        ndarray
            Global points (Ndim x Nedges x Nqp).

        """
        Y, tind = self.edge_reference(X, eind)
        return self.F(Y, tind=tind)

    def detDH(self,
              X: ndarray,
              eind: Optional[ndarray] = None) -> ndarray:
        """The length scaling of H, i.e. the norm of its derivative.

        Parameters
        ----------
        X
            Local points on the reference edge (1 x Nqp).
        eind
            A set of edge indices.

        Returns
        -------
        ndarray
            Length scalings at global points (Nedges x Nqp).

        """
        return np.linalg.norm(self.DH(X, eind), axis=0)

    def edge_tangents(self,
                      X: ndarray,
                      eind: Optional[ndarray] = None) -> ndarray:
        """Unit tangent vectors pointing from ``mesh.edges[0]`` towards
        ``mesh.edges[1]``.

        Parameters
        ----------
        X
            Local points on the reference edge (1 x Nqp).
        eind
            A set of edge indices.

        Returns
        -------
        ndarray
            Unit tangents (Ndim x Nedges x Nqp).

        """
        T = self.DH(X, eind)
        return T / np.linalg.norm(T, axis=0)

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
