import numpy as np

from .element import Element
from .discrete_field import DiscreteField


class ElementH1(Element):
    """:math:`H^1`-conforming basis defined through a reference element."""

    def gbasis(self, mapping, X, i, tind=None):
        """Identity transformation."""
        phi, dphi, sign = self._oriented_lbasis(mapping, X, i, tind)
        invDF = mapping.invDF(X, tind)
        value = np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3]))
        if len(X.shape) == 2 and dphi.ndim == 2:
            grad = np.einsum('ijkl,il->jkl', invDF, dphi)
        elif len(X.shape) == 3 or dphi.ndim == 3:
            grad = np.einsum('ijkl,ikl->jkl', invDF, dphi)
        else:
            raise NotImplementedError
        if sign is not None:
            value = value * sign[:, None]
            grad *= sign[:, None]
        return (DiscreteField(value=value, grad=grad),)

    def lbasis(self, X, i):
        raise NotImplementedError
