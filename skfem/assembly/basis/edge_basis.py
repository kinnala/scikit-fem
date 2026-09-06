import logging
from typing import Optional, Tuple, Any

import numpy as np
from numpy import ndarray
from skfem.element import DiscreteField, Element
from skfem.mapping import Mapping
from skfem.mesh import Mesh
from skfem.refdom import RefLine

from .abstract_basis import AbstractBasis
from ..dofs import Dofs


logger = logging.getLogger(__name__)


class EdgeBasis(AbstractBasis):
    """For integrating over edges of the mesh."""

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 edges: Optional[Any] = None,
                 dofs: Optional[Dofs] = None,
                 side: int = 0,
                 disable_doflocs: bool = False):
        """Precomputed global basis on edges."""
        typestr = ("{}({}, {})".format(type(self).__name__,
                                       type(mesh).__name__,
                                       type(elem).__name__))
        logger.info("Initializing {}".format(typestr))
        super(EdgeBasis, self).__init__(
            mesh,
            elem,
            mapping,
            intorder,
            quadrature,
            RefLine,
            dofs,
            disable_doflocs,
        )

        # by default use boundary edges (for testing only)
        if edges is None:
            self.eind = self.mesh.boundary_edges()
        else:
            self.eind = edges

        # TODO fix the orientation
        # element corresponding to the edge
        # is the first matching element from e2t
        tmp = mesh.e2t[:, self.eind]
        self.tind = tmp.indices[tmp.indptr[:-1]]

        if len(self.eind) == 0:
            logger.warning("Initializing {} with no edges.".format(typestr))

        # edge refdom to global edge
        x = self.mapping.H(self.X, eind=self.eind)
        # global edge to refdom edge
        Y = self.mapping.invF(x, tind=self.tind)

        # TODO calculate tangents
        self.tangents = ...

        self.nelems = len(self.eind)

        self.basis = [self.elem.gbasis(self.mapping, Y, j, tind=self.tind)
                      for j in range(self.Nbfun)]

        self.dx = (np.abs(self.mapping.detDH(self.X, eind=self.eind))
                   * np.broadcast_to(self.W, (self.nelems, self.W.shape[-1])))
        logger.info("Initializing finished.")

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {
            'x': self.global_coordinates(),
            'h': self.mesh_parameters(),
            't': self.tangents,
        }

    def global_coordinates(self) -> DiscreteField:
        return DiscreteField(self.mapping.H(self.X, eind=self.eind))

    def mesh_parameters(self) -> DiscreteField:
        return DiscreteField(np.abs(self.mapping.detDH(self.X, self.eind)))

    def with_element(self, elem: Element) -> 'EdgeBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            edges=self.eind,
        )
