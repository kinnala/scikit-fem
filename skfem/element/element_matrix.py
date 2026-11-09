import numpy as np
from .element_hdiv import ElementHdiv
from .discrete_field import DiscreteField


class ElementMatrix(ElementHdiv):
    """Matrix Piola mapping."""

    def gbasis(self, mapping, X, i, tind=None):
        """Matrix Piola transformation."""
        phi, _, sign = self._oriented_lbasis(mapping, X, i, tind)
        DF = mapping.DF(X, tind)
        detDF = mapping.detDF(X, tind)
        weight = 1 / (np.abs(detDF) ** 2)
        if sign is not None:
            weight = weight * sign[:, None]
        if len(X.shape) == 2 and phi.ndim == 3:
            return (DiscreteField(
                value=np.einsum('ijkl,jal,bakl,kl->ibkl', DF, phi, DF,
                                weight),
            ),)
        elif len(X.shape) == 3 or phi.ndim == 4:
            return (DiscreteField(
                value=np.einsum('ijkl,jakl,bakl,kl->ibkl', DF, phi, DF,
                                weight),
            ),)
        raise NotImplementedError
