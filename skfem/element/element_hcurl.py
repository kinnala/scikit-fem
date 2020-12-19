import numpy as np
from .element import Element
from .discrete_field import DiscreteField


class ElementHcurl(Element):
    """:math:`H(curl)`-conforming basis defined through a reference element.

    Supports 3D meshes only.

    """

    def orient(self, mapping, i, tind=None):
        """Orientation based on the edge node indexing."""
        if tind is not None:
            # TODO fix
            raise NotImplementedError("TODO: fix tind support in ElementHcurl")
        t1 = [0, 1, 0, 0, 1, 2][i]
        t2 = [1, 2, 2, 3, 3, 3][i]
        return 1 - 2 * (mapping.mesh.t[t1] > mapping.mesh.t[t2])

    def gbasis(self, mapping, X, i, tind=None):
        """Contravariant Piola transformation."""
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        invDF = mapping.invDF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        return (DiscreteField(
            value=np.einsum('ijkl,il,k->jkl', invDF, phi, orient),
            curl=np.einsum('ijkl,jl,kl->ikl', DF, dphi,
                           1. / detDF * orient[:, None])
        ),)

    def lbasis(self, X, i):
        raise Exception("ElementHcurl.lbasis method not found.")
