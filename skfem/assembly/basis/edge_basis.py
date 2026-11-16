import logging
from typing import Optional, Tuple, Any

import numpy as np
from numpy import ndarray
from skfem.element import DiscreteField, Element
from skfem.mapping import Mapping
from skfem.generic_utils import OrientedBoundary
from skfem.mesh import Mesh
from skfem.refdom import RefLine

from .abstract_basis import AbstractBasis
from ..dofs import Dofs


logger = logging.getLogger(__name__)


class EdgeBasis(AbstractBasis):
    """For integrating over edges of a 3D mesh, e.g., along a path.

    Integrate the solution of a 3D problem along a line:

    >>> import numpy as np
    >>> from skfem import MeshTet, Basis, EdgeBasis, ElementTetP1, Functional
    >>> m = MeshTet().refined(3).with_paths({
    ...     'axis': lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0),
    ... })
    >>> u = Basis(m, ElementTetP1()).project(lambda x: x[2] ** 2)
    >>> ebasis = EdgeBasis(m, ElementTetP1(), edges='axis')
    >>> integral = Functional(lambda w: w['u']).assemble(ebasis, u=u)
    >>> print(round(float(integral), 4))
    0.3332

    The default parameters available in the forms are the global
    coordinates ``w.x``, the edge lengths ``w.h`` and the unit tangents
    ``w.t``.  The tangents point from ``mesh.edges[0]`` to ``mesh.edges[1]``
    unless the edges are oriented, see
    :meth:`~skfem.mesh.Mesh.edges_satisfying`.

    """

    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 edges: Optional[Any] = None,
                 dofs: Optional[Dofs] = None,
                 disable_doflocs: bool = False):
        """Precomputed global basis on edges.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh` with
            three-dimensional elements.
        elem
            An object of type :class:`~skfem.element.Element`.
        mapping
            An object of type :class:`skfem.mapping.Mapping`. If `None`, uses
            `mesh.mapping`.
        intorder
            Optional integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature. Not used if `quadrature`
            is specified.
        quadrature
            Optional tuple of quadrature points and weights.
        edges
            A subset of edge indices or anything that can be passed to
            :meth:`~skfem.mesh.Mesh.normalize_edges`, e.g., the name of a
            path.  By default, the edges of all paths in ``mesh.paths``.
        dofs
            Optional :class:`~skfem.assembly.Dofs` object.
        disable_doflocs
            If `True`, the computation of global DOF locations is
            disabled.  This may save memory on large meshes if DOF
            locations are not required.

        """
        typestr = ("{}({}, {})".format(type(self).__name__,
                                       type(mesh).__name__,
                                       type(elem).__name__))
        logger.info("Initializing {}".format(typestr))
        if mesh.dim() != 3:
            raise ValueError("EdgeBasis requires a mesh with "
                             "three-dimensional elements; use FacetBasis "
                             "for the edges of a two-dimensional mesh.")
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

        self.eind = mesh.normalize_edges(edges)

        if len(self.eind) == 0:
            logger.warning("Initializing {} with no edges.".format(typestr))

        # reference edge to reference element; one element per edge
        Y, self.tind = self.mapping.edge_reference(self.X, eind=self.eind)

        # unit tangents, flipped according to the orientation
        tangents = self.mapping.edge_tangents(self.X, eind=self.eind)
        if isinstance(self.eind, OrientedBoundary):
            tangents = tangents * (1. - 2. * self.eind.ori)[None, :, None]
        self.tangents = DiscreteField(value=tangents)

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
        if self._global_coordinates is None:
            self._global_coordinates = DiscreteField(
                self.mapping.H(self.X, eind=self.eind)
            )
        return self._global_coordinates

    def mesh_parameters(self) -> DiscreteField:
        if self._mesh_parameters is None:
            self._mesh_parameters = DiscreteField(
                np.abs(self.mapping.detDH(self.X, eind=self.eind))
            )
        return self._mesh_parameters

    def with_element(self, elem: Element) -> 'EdgeBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            edges=self.eind,
        )

    def split_bases(self):
        from skfem.element import ElementComposite, ElementVector
        if isinstance(self.elem, ElementComposite):
            return [self.with_element(e) for e in self.elem.elems]
        elif isinstance(self.elem, ElementVector):
            return [self.with_element(self.elem.elem)
                    for _ in range(self.elem.dim)]
        return [self]

    def project(self, interp, edges=None, dtype=None):
        """Perform :math:`L^2` projection onto the basis on the edges.

        The values of the remaining DOFs are zero.

        Parameters
        ----------
        interp
            An object of type :class:`~skfem.element.DiscreteField` which is a
            function (to be projected) evaluated at global quadrature points
            on the edges.  If a function is given, then
            :class:`~skfem.element.DiscreteField` is created by passing
            an array of global quadrature point locations to the function.
        edges
            Optionally perform the projection on a subset of edges.
        dtype
            Set to `np.complex64` or similar to use complex numbers.

        """
        from skfem.utils import solve, condense

        M, f = self._projection(interp, dtype=dtype)

        return solve(*condense(M, f, I=self.get_dofs(
            edges=self.eind if edges is None else edges
        )))
