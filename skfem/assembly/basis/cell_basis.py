from typing import Callable, Optional, Tuple

import numpy as np
from numpy import ndarray
from scipy.sparse import coo_matrix
from skfem.element import DiscreteField, Element
from skfem.mapping import Mapping
from skfem.mesh import Mesh

from .abstract_basis import AbstractBasis


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
                 elements: Optional[ndarray] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None):
        """Combine :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`
        into a set of precomputed global basis functions.

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

        """
        super(CellBasis, self).__init__(mesh,
                                        elem,
                                        mapping,
                                        intorder,
                                        quadrature,
                                        mesh.refdom)

        self.basis = [self.elem.gbasis(self.mapping, self.X, j, tind=elements)
                      for j in range(self.Nbfun)]

        if elements is None:
            self.nelems = mesh.nelements
        else:
            self.nelems = len(elements)
        self.tind = elements

        self.dx = (np.abs(self.mapping.detDF(self.X, tind=elements))
                   * np.tile(self.W, (self.nelems, 1)))

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {'x': self.global_coordinates(),
                'h': self.mesh_parameters()}

    def global_coordinates(self) -> DiscreteField:
        return DiscreteField(self.mapping.F(self.X, tind=self.tind))

    def mesh_parameters(self) -> DiscreteField:
        return DiscreteField(np.abs(self.mapping.detDF(self.X, self.tind))
                             ** (1. / self.mesh.dim()))

    def refinterp(self,
                  y: ndarray,
                  nrefs: int = 1,
                  Nrefs: Optional[int] = None) -> Tuple[Mesh, ndarray]:
        """Refine and interpolate (for plotting)."""
        if Nrefs is not None:
            nrefs = Nrefs  # for backwards compatibility
        # mesh reference domain, refine and take the vertices
        meshclass = type(self.mesh)
        m = meshclass.init_refdom().refined(nrefs)
        X = m.p

        # map vertices to global elements
        x = self.mapping.F(X)

        # interpolate some previous discrete function at the vertices
        # of the refined mesh
        w = 0. * x[0]
        for j in range(self.Nbfun):
            basis = self.elem.gbasis(self.mapping, X, j)
            w += y[self.element_dofs[j]][:, None] * basis[0]

        # create connectivity for the new mesh
        nt = self.nelems
        t = np.tile(m.t, (1, nt))
        dt = np.max(t)
        t += (dt + 1) *\
            (np.tile(np.arange(nt), (m.t.shape[0] * m.t.shape[1], 1))
             .flatten('F')
             .reshape((-1, m.t.shape[0])).T)

        if X.shape[0] == 1:
            p = np.array([x.flatten()])
        else:
            p = x[0].flatten()
            for itr in range(len(x) - 1):
                p = np.vstack((p, x[itr + 1].flatten()))

        M = meshclass(p, t)

        return M, w.flatten()

    def probes(self, x: ndarray) -> coo_matrix:
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
                self.elem.gbasis(self.mapping, pts, k, tind=cells)[0][0]
                for k in range(self.Nbfun)
            ]
        ).flatten()
        return coo_matrix(
            (
                phis,
                (
                    np.tile(np.arange(x.shape[1]), self.Nbfun),
                    self.element_dofs[:, cells].flatten(),
                ),
            ),
            shape=(x.shape[1], self.N),
        )

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
            return self.probes(x) @ y

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
