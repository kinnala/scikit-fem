import numpy as np
from .element import Element


class ElementHdiv(Element):
    order = (1, 0)

    def orient(self, mapping, i, tind=None):
        if tind is not None:
            # TODO fix
            raise NotImplementedError("TODO: fix tind support in ElementHdiv")
        return -1 + 2*(mapping.mesh.f2t[0, mapping.mesh.t2f[i, :]]\
                       == np.arange(mapping.mesh.t.shape[1]))

    def gbasis(self, mapping, X, i, tind=None):
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        return np.einsum('ijkl,jl,kl->ikl',
                         DF,
                         phi,
                         1 / np.abs(detDF) * orient[:, None]),\
                         dphi / (np.abs(detDF) * orient[:, None])

    def lbasis(self, X, i):
        raise Exception("ElementHdiv lbasis method not found.")
