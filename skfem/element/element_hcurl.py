import numpy as np
from .element import Element
from .discrete_field import DiscreteField


class ElementHcurl(Element):
    """:math:`H(curl)`-conforming basis defined through a reference element."""

    def orient(self, mapping, i, tind=None):
        """Orientation based on the edge node indexing."""
        if tind is not None:
            # TODO fix
            raise NotImplementedError("TODO: fix tind support in ElementHcurl")
        if mapping.mesh.dim() == 3 and mapping.mesh.t.shape[0] == 4:
            t1 = [0, 1, 0, 0, 1, 2][i]
            t2 = [1, 2, 2, 3, 3, 3][i]
        elif mapping.mesh.dim() == 2 and mapping.mesh.t.shape[0] == 3:
            t1 = [0, 1, 0][i]
            t2 = [1, 2, 2][i]
        else:
            raise NotImplementedError("The element type not supported.")
        return 1 - 2 * (mapping.mesh.t[t1] > mapping.mesh.t[t2])

    def gbasis(self, mapping, X, i, tind=None):
        """Covariant Piola transformation."""
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        invDF = mapping.invDF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        if mapping.mesh.dim() == 3:
            return (DiscreteField(
                value=np.einsum('ijkl,il,k->jkl', invDF, phi, orient),
                curl=np.einsum('ijkl,jl,kl->ikl', DF, dphi,
                               1. / detDF * orient[:, None]),
            ),)
        else:
            return (DiscreteField(
                # TODO orientation is automatically correct for triangular
                # meshes, check that this works for quadrilateral meshes
                value=np.einsum('ijkl,il,k->jkl', invDF, phi, orient),
                curl=dphi / np.abs(detDF) * orient[:, None],
            ),)

    def lbasis(self, X, i):
        raise NotImplementedError
