from typing import Dict, Optional

import numpy as np
from numpy import ndarray

from skfem.quadrature import get_quadrature

from .global_basis import GlobalBasis
from .interior_basis import InteriorBasis


class MortarBasis(GlobalBasis):
    """Global basis functions evaluated at integration points on the mortar
    boundary. """
    def __init__(self,
                 mesh,
                 elem,
                 mapping,
                 intorder: Optional[int] = None,
                 side: int = 0):
        super(MortarBasis, self).__init__(mesh, elem, mapping, intorder)

        self.ib1 = InteriorBasis(mesh.mesh1, elem)
        self.ib2 = InteriorBasis(mesh.mesh2, elem)

        self.X, self.W = get_quadrature(self.brefdom, self.intorder)

        self.find = np.nonzero(self.mesh.f2t[1, :] != -1)[0]
        self.tind = self.mesh.f2t[side, self.find]

        # boundary refdom to global facet
        x = self.mapping.G(self.X, find=self.find)
        # global facet to refdom facet
        Y = self.mapping.invF(x, tind=self.tind)

        self.normals = np.repeat(mesh.normals[:, :, None], len(self.W), axis=2)

        self.nelems = len(self.find)

        self.basis = [self.elem.gbasis(self.mapping, Y, j, self.tind)
                      for j in range(self.Nbfun)]

        self.dx = np.abs(self.mapping.detDG(self.X, find=self.find)) *\
            np.tile(self.W, (self.nelems, 1))

        if side == 0:
            self.element_dofs = self.ib1.element_dofs[:, self.tind]
        elif side == 1:
            self.element_dofs = self.ib2.element_dofs[:, self.tind - mesh.mesh1.t.shape[1]] + self.ib1.N

        self.N = self.ib1.N + self.ib2.N

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {'x':self.global_coordinates(),
                'h':self.mesh_parameters(),
                'n':self.normals}
    
    def global_coordinates(self) -> ndarray:
        return self.mapping.G(self.X, find=self.find)

    def mesh_parameters(self) -> ndarray:
        if self.mesh.dim() == 1:
            return np.array([0.0])
        else:
            return np.abs(self.mapping.detDG(self.X, self.find)) ** (1.0 / (self.mesh.dim() - 1))
