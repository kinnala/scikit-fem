from typing import Optional, Callable, Tuple

import numpy as np
from numpy import ndarray

from skfem.element import Element, DiscreteField
from skfem.mapping import Mapping
from skfem.mesh import Mesh
from skfem.quadrature import get_quadrature
from .basis import Basis


class InteriorBasis(Basis):
    """Basis functions evaluated at quadrature points inside the elements.

    :class:`~skfem.assembly.InteriorBasis` object is a combination of
    :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element`:

    >>> from skfem import *
    >>> m = MeshTri.init_symmetric()
    >>> e = ElementTriP1()
    >>> basis = InteriorBasis(m, e)

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

        super(InteriorBasis, self).__init__(mesh, elem, mapping)

        if quadrature is not None:
            self.X, self.W = quadrature
        else:
            self.X, self.W = get_quadrature(
                self.refdom,
                intorder if intorder is not None else 2 * self.elem.maxdeg
            )

        self.basis = [self.elem.gbasis(self.mapping, self.X, j, tind=elements)
                      for j in range(self.Nbfun)]

        if elements is None:
            self.nelems = mesh.t.shape[1]
            self.tind = np.arange(self.nelems, dtype=np.int64)
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
                  interp: ndarray,
                  Nrefs: Optional[int] = 1) -> Tuple[Mesh, ndarray]:
        """Refine and interpolate (for plotting)."""
        # mesh reference domain, refine and take the vertices
        meshclass = type(self.mesh)
        m = meshclass.init_refdom().refined(Nrefs)
        X = m.p

        # map vertices to global elements
        x = self.mapping.F(X)

        # interpolate some previous discrete function at the vertices
        # of the refined mesh
        w = 0. * x[0]
        for j in range(self.Nbfun):
            basis = self.elem.gbasis(self.mapping, X, j)
            w += interp[self.element_dofs[j]][:, None] * basis[0]

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

        M = meshclass(p, t, validate=False)

        return M, w.flatten()

    def interpolator(self, y: ndarray) -> Callable[[ndarray], ndarray]:
        """Return a function handle, which can be used for finding
        pointwise values of the given solution vector."""

        finder = self.mesh.element_finder(mapping=self.mapping)

        def interpfun(x):
            tris = finder(*x)
            pts = self.mapping.invF(x[:, :, np.newaxis], tind=tris)
            w = np.zeros(x.shape[1])
            for k in range(self.Nbfun):
                phi = self.elem.gbasis(self.mapping, pts, k, tind=tris)[0]
                w += y[self.element_dofs[k, tris]] * phi[0].flatten()
            return w

        return interpfun
