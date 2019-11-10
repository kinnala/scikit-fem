import numpy as np
from .element import Element


class ElementHcurl(Element):
    """Note: only 3D support. Piola transformation
    is different in 2D."""
    
    order = (1, 1)

    def orient(self, mapping, i, tind=None):
        if tind is not None:
            # TODO fix
            raise NotImplementedError("TODO: fix tind support in ElementHcurl")
        t1 = [0, 1, 0, 0, 1, 2][i]
        t2 = [1, 2, 2, 3, 3, 3][i]
        return 1 - 2*(mapping.mesh.t[t1, :] > mapping.mesh.t[t2, :])

    def gbasis(self, mapping, X, i, tind=None):
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        invDF = mapping.invDF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        return np.einsum('ijkl,il,k->jkl', invDF, phi, orient),\
               np.einsum('ijkl,jl,kl->ikl', DF, dphi, 1/detDF*orient[:, None])

    def lbasis(self, X, i):
        raise Exception("ElementHcurl lbasis method not found.")
