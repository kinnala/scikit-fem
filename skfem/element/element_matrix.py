import numpy as np
from .element_hdiv import ElementHdiv
from .discrete_field import DiscreteField


class ElementMatrix(ElementHdiv):
    """Matrix Piola mapping."""

    def gbasis(self, mapping, X, i, tind=None):
        """Matrix Piola transformation."""
        phi, _ = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        detDF = mapping.detDF(X, tind)
        if len(X.shape) == 2:
            return (DiscreteField(
                value=np.einsum('ijkl,jal,bakl,kl->ibkl', DF, phi, DF,
                                1 / (np.abs(detDF) ** 2)),
            ),)
        elif len(X.shape) == 3:
            return (DiscreteField(
                value=np.einsum('ijkl,jakl,bakl,kl->ibkl', DF, phi, DF,
                                1 / (np.abs(detDF) ** 2)),
            ),)
        raise NotImplementedError
