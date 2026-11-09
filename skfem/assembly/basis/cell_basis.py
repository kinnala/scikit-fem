import logging
from typing import Callable, Optional, Tuple, Any

import numpy as np
from numpy import ndarray
from skfem.element import DiscreteField, Element
from skfem.mapping import Mapping, MappingManifold
from skfem.mesh import Mesh
from skfem.generic_utils import Removed

from scipy.sparse import coo_matrix

from .abstract_basis import AbstractBasis
from ..dofs import Dofs


logger = logging.getLogger(__name__)


class CellBasis(AbstractBasis):
    """For fields defined inside the domain.

    :class:`~skfem.assembly.CellBasis` object is a combination of
    :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`.

    >>> from skfem import *
    >>> m = MeshTri.init_symmetric()
    >>> e = ElementTriP1()
    >>> basis = CellBasis(m, e)

    The resulting objects are used in the assembly.

    >>> from skfem.models.poisson import laplace
    >>> K = asm(laplace, basis)
    >>> K.shape
    (5, 5)

    """
    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 elements: Optional[Any] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None,
                 dofs: Optional[Dofs] = None,
                 disable_doflocs: bool = False):
        """Combine :class:`~skfem.mesh.Mesh` and
        :class:`~skfem.element.Element` into a set of precomputed global basis
        functions.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh`.
        elem
            An object of type :class:`~skfem.element.Element`.
        mapping
            An object of type :class:`skfem.mapping.Mapping`. If `None`, uses
            `mesh.mapping`.
        intorder
            Optional integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature. Not used if `quadrature`
            is specified.
        elements
            Optional subset of element indices.
        quadrature
            Optional tuple of quadrature points and weights.
        dofs
            Optional :class:`~skfem.assembly.Dofs` object.
        disable_doflocs
            If `True`, the computation of global DOF locations is
            disabled.  This may save memory on large meshes if DOF
            locations are not required.

        """
        logger.info("Initializing {}({}, {})".format(type(self).__name__,
                                                     type(mesh).__name__,
                                                     type(elem).__name__))
        super(CellBasis, self).__init__(
            mesh,
            elem,
            mapping,
            intorder,
            quadrature,
            mesh.refdom,
            dofs,
            disable_doflocs,
        )

        if elements is None:
            self.tind = None
            self.nelems = mesh.nelements
        else:
            self.tind = mesh.normalize_elements(elements)
            self.nelems = len(self.tind)

        self.basis = [self.elem.gbasis(self.mapping, self.X, j, tind=self.tind)
                      for j in range(self.Nbfun)]

        self.dx = (np.abs(self.mapping.detDF(self.X, tind=self.tind))
                   * np.broadcast_to(self.W, (self.nelems, self.W.shape[-1])))
        logger.info("Initializing finished.")

    @property
    def _base_tensor_order(self):

        loc_pts = np.zeros((self.elem.dim, 1))[:, :, np.newaxis]
        base_obj = self.elem.gbasis(
            self.mapping,
            loc_pts,
            0,
            tind=np.array([0], dtype=np.int32)
        )

        if len(base_obj) > 1:
            raise NotImplementedError

        return base_obj[0].shape[:-2]

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        params = {'x': self.global_coordinates(),
                  'h': self.mesh_parameters()}
        if isinstance(self.mapping, MappingManifold):
            # embedded curves and surfaces
            if self.mesh.dim() == 1:
                params['t'] = DiscreteField(
                    self.mapping.tangents(self.X, tind=self.tind))
            if self.mapping.gdim - self.mesh.dim() == 1:
                params['n'] = DiscreteField(
                    self.mapping.cell_normals(self.X, tind=self.tind))
        return params

    def global_coordinates(self) -> DiscreteField:
        if self._global_coordinates is None:
            self._global_coordinates = DiscreteField(
                self.mapping.F(self.X, tind=self.tind)
            )
        return self._global_coordinates

    def mesh_parameters(self) -> DiscreteField:
        if self._mesh_parameters is None:
            self._mesh_parameters = DiscreteField(
                np.abs(self.mapping.detDF(self.X, self.tind))
                ** (1. / self.mesh.dim()))
        return self._mesh_parameters

    def refinterp(self,
                  y: ndarray,
                  nrefs: int = 1,
                  Nrefs: Optional[int] = None) -> Tuple[Mesh, ndarray]:
        """Refine and interpolate (for plotting).

        Thin wrapper around :func:`skfem.visuals.data.sample` which returns
        a discontinuous linear mesh of the subdivided elements and the values
        of ``y`` at its vertices.  Vector- and tensor-valued fields are
        returned component by component, i.e. the length of the output is a
        multiple of the number of vertices.

        """
        from ...visuals.data import sample
        if Nrefs is not None:
            nrefs = Nrefs  # for backwards compatibility
        # in 1D, shrinking the reference points makes sorting preserve
        # correct order of duplicate nodes in skfem.visuals.matplotlib
        data = sample(self, y, nrefs=nrefs, _shrink=self.mesh.dim() == 1)
        w = data.point_data['u']
        return data.to_mesh(), np.ascontiguousarray(w.T).flatten()

    def probes(self, x: ndarray):
        """Return matrix which acts on a solution vector to find its values
        on points `x`.

        The product of this with a finite element function vector is like the
        result of assembling a `Functional` and it can be thought of as the
        matrix of inner products of the test functions of the basis with Dirac
        deltas at `x` but because its action is concentrated at points it is
        not assembled with the usual quadratures.

        """
        cells = self.mesh.element_finder(mapping=self.mapping)(*x)
        pts = self.mapping.invF(x[:, :, np.newaxis], tind=cells)
        phis = np.array(
            [
                self.elem.gbasis(self.mapping, pts, k, tind=cells)[0]
                for k in range(self.Nbfun)
            ]
        ).flatten()
        # number of components of a base functions
        comp = int(np.prod(self._base_tensor_order))
        # row indices
        rows = np.tile(np.arange(comp * x.shape[1],
                                 dtype=np.int32), self.Nbfun)
        # col indices
        cols = self.element_dofs[:, np.tile(cells, comp)].flatten()
        # shape
        sh = (comp * x.shape[1], self.N)
        return coo_matrix((phis, (rows, cols,)), shape=sh)

    def point_source(self, x: ndarray) -> ndarray:
        """Return right-hand side vector for unit source at `x`,

        i.e. the vector of inner products of a Dirac delta at `x`
        with the test functions of the basis.

        This is like what is obtained by assembling a `LinearForm`
        but because its action is concentrated at points it is not
        assembled with the usual quadratures.

        """
        return self.probes(x[:, None]).toarray()[0]

    def interpolator(self, y: ndarray) -> Callable[[ndarray], ndarray]:
        """Return a function handle, which can be used for finding
        values of the given solution vector `y` on given points."""

        def interpfun(x: ndarray) -> ndarray:
            # reshape to 2-array to support trailing axes
            # this is useful, e.g., to pass interpfun to Basis.project
            shape = None
            if len(x.shape) > 2:
                shape = x.shape
                x = x.reshape(shape[0], -1)
            out = self.probes(x) @ y
            # reshape output for tensor like base functions
            if len(self._base_tensor_order) > 0:
                out = out.reshape(self._base_tensor_order + (x.shape[1],))
            # reshape output back to original shape
            if shape is not None:
                return out.reshape(self._base_tensor_order + shape[1:])
            return out

        return interpfun

    def with_element(self, elem: Element) -> 'CellBasis':
        """Return a similar basis using a different element."""
        return type(self)(
            self.mesh,
            elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            elements=self.tind,
        )

    def with_elements(self, elements: Optional[Any] = None) -> 'CellBasis':
        """Return a similar basis on a subset of element indices."""
        return type(self)(
            self.mesh,
            self.elem,
            mapping=self.mapping,
            quadrature=self.quadrature,
            elements=elements,
        )

    def boundary(self,
                 facets: Optional[Any] = None,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None):
        """Return corresponding :class:`~skfem.assembly.basis.FacetBasis`.

        Parameters
        ----------
        facets
            Anything that can be passed to ``FacetBasis(..., facets=facets)``.
        intorder
            Optionally, specify integration order.
        quadrature
            Optionally, specify quadrature.

        """
        from skfem.assembly.basis.facet_basis import FacetBasis
        if self.tind is not None:
            raise NotImplementedError("Boundary of subdomain not supported.")
        return FacetBasis(
            self.mesh,
            self.elem,
            mapping=self.mapping,
            facets=facets,
            intorder=intorder,
            quadrature=quadrature,
        )

    def project(self, interp, elements=None, dtype=None):
        """Perform :math:`L^2` projection onto the basis.

        See :ref:`l2proj` for more information.

        Parameters
        ----------
        interp
            An object of type :class:`~skfem.element.DiscreteField` which is a
            function (to be projected) evaluated at global quadrature points.
            If a function is given, then :class:`~skfem.element.DiscreteField`
            is created by passing an array of global quadrature point locations
            to the function.
        elements
            Optionally perform the projection on a subset of elements.  The
            values of the remaining DOFs are zero.
        dtype
            Set to `np.complex64` or similar to use complex numbers.

        """
        from skfem.utils import solve, condense

        M, f = self._projection(interp, dtype=dtype)

        if elements is not None:
            return solve(*condense(M, f, I=self.get_dofs(elements=elements)))
        elif self.tind is not None:
            return solve(*condense(M, f, I=self.get_dofs(elements=self.tind)))
        return solve(M, f)

    find_dofs = Removed(
        version="8.0.0", era="pre-4.0",
        message=("Use Basis.get_dofs, which returns a DofsView directly "
                 "rather than a dict:\n"
                 "    D = basis.get_dofs('left')\n"
                 "The tag is passed positionally, not as a dict of lambdas."))
