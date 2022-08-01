import numpy as np
from .element import Element
from .discrete_field import DiscreteField


class ElementHdiv(Element):
    """:math:`H(div)`-conforming basis defined through a reference element."""

    def orient(self, mapping, i, tind=None):
        """Orientation based on facet-to-triangle indexing."""
        if tind is None:
            tind = slice(None)
        ix = int(i / self.facet_dofs)
        if ix >= self.refdom.nfacets:
            # interior dofs: no need for orientation
            # TODO support edge DOFs
            return np.ones(len(np.arange(mapping.mesh.t.shape[1])[tind]),
                           dtype=np.int64)
        ori = -1 + 2 * (mapping.mesh.f2t[0, mapping.mesh.t2f[ix]]
                        == np.arange(mapping.mesh.t.shape[1]))
        return ori[tind]

    def gbasis(self, mapping, X, i, tind=None):
        """Covariant Piola transformation."""
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        if len(X.shape) == 2:
            return (DiscreteField(
                value=np.einsum('ijkl,jl,kl->ikl', DF, phi,
                                1. / np.abs(detDF) * orient[:, None]),
                div=dphi / (np.abs(detDF) * orient[:, None])
            ),)
        elif len(X.shape) == 3:
            return (DiscreteField(
                value=np.einsum('ijkl,jkl,kl->ikl', DF, phi,
                                1. / np.abs(detDF) * orient[:, None]),
                div=dphi / (np.abs(detDF) * orient[:, None])
            ),)
        raise NotImplementedError

    def lbasis(self, X, i):
        raise NotImplementedError
