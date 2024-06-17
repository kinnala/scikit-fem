import numpy as np
from .element import Element
from .discrete_field import DiscreteField


class ElementHcurl(Element):
    """:math:`H(curl)`-conforming basis defined through a reference element."""

    def orient(self, mapping, i, tind=None):
        """Orientation based on the edge node indexing."""
        if tind is None:
            tind = slice(None)
        divide_by = (self.facet_dofs
                     if mapping.mesh.dim() == 2
                     else self.edge_dofs)
        ix = int(i / divide_by)
        if mapping.mesh.dim() == 2 and ix >= self.refdom.nfacets:
            # no orientation required for interior DOFs => return 1
            ori = np.ones(mapping.mesh.t.shape[1], dtype=np.int32)
            return ori[tind]
        if mapping.mesh.dim() == 3:
            t1, t2 = mapping.mesh.refdom.edges[ix]
        elif mapping.mesh.dim() == 2:
            t1, t2 = mapping.mesh.refdom.facets[ix]
        else:
            raise NotImplementedError("The element type not supported.")
        ori = 1 - 2 * (mapping.mesh.t[t1] > mapping.mesh.t[t2])
        return ori[tind]

    def gbasis(self, mapping, X, i, tind=None):
        """Covariant Piola transformation."""
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        invDF = mapping.invDF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        if mapping.mesh.dim() == 3:
            if len(X.shape) == 2:
                return (DiscreteField(
                    value=np.einsum('ijkl,il,k->jkl', invDF, phi, orient),
                    curl=np.einsum('ijkl,jl,kl->ikl', DF, dphi,
                                   1. / detDF * orient[:, None]),
                ),)
            elif len(X.shape) == 3:
                return (DiscreteField(
                    value=np.einsum('ijkl,ikl,k->jkl', invDF, phi, orient),
                    curl=np.einsum('ijkl,jkl,kl->ikl', DF, dphi,
                                   1. / detDF * orient[:, None]),
                ),)
        else:
            if len(X.shape) == 2:
                return (DiscreteField(
                    value=np.einsum('ijkl,il,k->jkl', invDF, phi, orient),
                    curl=dphi / detDF * orient[:, None],
                ),)
            elif len(X.shape) == 3:
                return (DiscreteField(
                    value=np.einsum('ijkl,ikl,k->jkl', invDF, phi, orient),
                    curl=dphi / detDF * orient[:, None],
                ),)
        raise NotImplementedError

    def lbasis(self, X, i):
        raise NotImplementedError
