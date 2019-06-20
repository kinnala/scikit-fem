import numpy as np
from .element import Element


class ElementH1(Element):
    order = (0, 1)

    def gbasis(self, mapping, X, i, tind=None):
        phi, dphi = self.lbasis(X, i)
        invDF = mapping.invDF(X, tind)
        if len(X.shape) == 2:
            return np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])),\
                   np.einsum('ijkl,il->jkl', invDF, dphi)
        elif len(X.shape) == 3:
            return np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])),\
                   np.einsum('ijkl,ikl->jkl', invDF, dphi)

    def lbasis(self, X, i):
        raise Exception("ElementH1 lbasis method not found.")
