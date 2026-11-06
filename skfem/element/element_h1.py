import numpy as np

from .element import Element
from .discrete_field import DiscreteField


class ElementH1(Element):
    """:math:`H^1`-conforming basis defined through a reference element."""

    def gbasis(self, mapping, X, i, tind=None):
        """Identity transformation."""
        phi, dphi = self._oriented_lbasis(mapping, X, i, tind)
        invDF = mapping.invDF(X, tind)
        if len(X.shape) == 2 and dphi.ndim == 2:
            return (DiscreteField(
                value=np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])),
                grad=np.einsum('ijkl,il->jkl', invDF, dphi)
            ),)
        elif len(X.shape) == 3 or dphi.ndim == 3:
            return (DiscreteField(
                value=np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])),
                grad=np.einsum('ijkl,ikl->jkl', invDF, dphi)
            ),)
        raise NotImplementedError

    def lbasis(self, X, i):
        raise NotImplementedError

    def reversed_facet_dof(self, j):
        """Return the facet DOF replacing the j'th one on a reversed facet.

        Elements with several DOFs per facet, which depend on the direction
        of the facet, override this.  The facet of an element is reversed if
        its local direction is opposite to the direction of increasing global
        vertex indices.  Returns a tuple ``(j', sign)`` meaning that the basis
        function corresponding to the local facet DOF ``j'``, multiplied by
        ``sign``, is used.  The default ``None`` means no reordering.

        """
        return None

    def _oriented_lbasis(self, mapping, X, i, tind=None):
        """Evaluate lbasis and reorder the DOFs of the reversed facets."""
        phi, dphi = self.lbasis(X, i)
        offset = self.refdom.nnodes * self.nodal_dofs
        if (self.dim != 2
                or self.facet_dofs < 2
                or not offset <= i < offset + self.refdom.nfacets
                * self.facet_dofs):
            return phi, dphi
        facet, j = divmod(i - offset, self.facet_dofs)
        rev = self.reversed_facet_dof(j)
        if rev is None:
            return phi, dphi
        t = mapping.mesh.t if tind is None else mapping.mesh.t[:, tind]
        a, b = self.refdom.facets[facet]
        reverse = t[a] > t[b]
        if not reverse.any():
            return phi, dphi
        phi_r, dphi_r = self.lbasis(X, offset + facet * self.facet_dofs
                                    + rev[0])
        phi_r, dphi_r = rev[1] * phi_r, rev[1] * dphi_r
        if phi.ndim == 1:
            # evaluate per element
            phi, phi_r = phi[None], phi_r[None]
            dphi, dphi_r = dphi[:, None], dphi_r[:, None]
        return (np.where(reverse[:, None], phi_r, phi),
                np.where(reverse[None, :, None], dphi_r, dphi))
