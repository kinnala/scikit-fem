import numpy as np
from .element import Element


class ElementVectorH1(Element):
    order = (1, 2)

    def __init__(self, elem):
        self.dim = elem.dim
        self.elem = elem

        self.nodal_dofs = self.elem.nodal_dofs * self.dim
        self.facet_dofs = self.elem.facet_dofs * self.dim
        self.interior_dofs = self.elem.interior_dofs * self.dim
        self.edge_dofs = self.elem.edge_dofs * self.dim

        self.dofnames = [i + "^" + str(j + 1)
                         for i in elem.dofnames
                         for j in range(self.dim)]
        self.maxdeg = elem.maxdeg

    def gbasis(self, mapping, X, i, tind=None):
        ind = int(np.floor(float(i) / float(self.dim)))
        n = i - self.dim*ind
        phi, dphi = self.elem.gbasis(mapping, X, ind, tind)
        u = np.zeros((self.dim,) + phi.shape)
        du = np.zeros((self.dim,) + dphi.shape)
        u[n] = phi
        du[n] = dphi
        return u, du
