import numpy as np

from .element import Element
from .discrete_field import DiscreteField


class ElementH1(Element):
    """:math:`H^1`-conforming basis defined through a reference element."""

    def gbasis(self, mapping, X, i, tind=None):
        """Identity transformation."""
        phi, dphi = self.lbasis(X, i)
        invDF = mapping.invDF(X, tind)
        if len(X.shape) == 2:
            return (DiscreteField(
                value=np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])),
                grad=np.einsum('ijkl,il->jkl', invDF, dphi)
            ),)
        elif len(X.shape) == 3:
            return (DiscreteField(
                value=np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])),
                grad=np.einsum('ijkl,ikl->jkl', invDF, dphi)
            ),)

    def lbasis(self, X, i):
        raise Exception("ElementH1.lbasis method not found.")
