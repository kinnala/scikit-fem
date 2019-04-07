import numpy as np

from typing import Optional, Callable, Tuple

from numpy import ndarray

from skfem.quadrature import get_quadrature

from skfem.mesh import Mesh
from skfem.element import Element
from skfem.mapping import Mapping
from .global_basis import GlobalBasis


class InteriorBasis(GlobalBasis):
    """Global basis functions evaluated at integration points inside the
    elements.

    Attributes
    ----------
    phi : ndarray
        Global basis functions at global quadrature points.
    dphi : ndarray
        Global basis function derivatives at global quadrature points.
    X : ndarray
        Local quadrature points (Ndim x Nqp).
    W : ndarray
        Local quadrature weights (Nqp).
    nelems : int
    dx : ndarray
        Can be used in computing global integrals elementwise (Nelems
        x Nqp).  For example, np.sum(u**2*dx, axis=1) where u is also
        a numpy array of size Nelems x Nqp.
    mapping : skfem.mapping.Mapping
    elem : skfem.element.Element
    Nbfun : int
        Number of basis functions.
    intorder : int
        Integration order.
    dim : int
        Dimension of the problem.
    nt : int
        Number of triangles.
    mesh : skfem.mesh.Mesh
    refdom : string
    brefdom : string

    Examples
    --------
    :class:`~skfem.assembly.InteriorBasis` object is a combination of
    :class:`~skfem.mesh.Mesh`, :class:`~skfem.element.Element`, and
    :class:`~skfem.mapping.Mapping`:

    >>> from skfem import *
    >>> from skfem.models.poisson import laplace
    >>> m = MeshTri.init_symmetric()
    >>> e = ElementTriP1()
    >>> ib = InteriorBasis(m, e, MappingAffine(m))

    The resulting objects are used in the assembly.

    >>> K = asm(laplace, ib)
    >>> K.shape
    (5, 5)

    """
    def __init__(self,
                 mesh: Mesh,
                 elem: Element,
                 mapping: Optional[Mapping] = None,
                 intorder: Optional[int] = None,
                 elements: Optional[ndarray] = None):
        """Combine :class:`~skfem.mesh.Mesh` and :class:`~skfem.element.Element` into a
        set of precomputed global basis functions.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh`.
        elem
            An object of type :class:`~skfem.element.Element`.
        mapping
            An object of type :class:`~skfem.mapping.Mapping`.
        intorder
            Optional integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature.
        elements
            Optional subset of element indices.

        """
        super(InteriorBasis, self).__init__(mesh, elem, mapping, intorder)

        self.X, self.W = get_quadrature(self.refdom, self.intorder)

        self.basis = [self.elem.gbasis(self.mapping, self.X, j, tind=elements)
                      for j in range(self.Nbfun)]

        if elements is None:
            self.nelems = self.mesh.t.shape[1]
            self.tind = np.arange(self.nelems, dtype=np.int64)
        else:
            self.nelems = len(elements)
            self.tind = elements

        self.dx = np.abs(self.mapping.detDF(self.X, tind=elements)) *\
            np.tile(self.W, (self.nelems, 1))

        self.element_dofs = self.element_dofs[:, self.tind]

    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {'x': self.global_coordinates(),
                'h': self.mesh_parameters()}

    def global_coordinates(self) -> ndarray:
        return self.mapping.F(self.X, tind=self.tind)

    def mesh_parameters(self) -> ndarray:
        return np.abs(self.mapping.detDF(self.X, self.tind)) ** (1.0 / self.mesh.dim())

    def refinterp(self,
                  interp: ndarray,
                  Nrefs: Optional[int] = 1) -> Tuple[Mesh, ndarray]:
        """Refine and interpolate (for plotting)."""
        # mesh reference domain, refine and take the vertices
        meshclass = type(self.mesh)
        m = meshclass.init_refdom()
        m.refine(Nrefs)
        X = m.p

        # map vertices to global elements
        x = self.mapping.F(X)

        # interpolate some previous discrete function at the vertices
        # of the refined mesh
        w = 0.0*x[0]

        for j in range(self.Nbfun):
            basis = self.elem.gbasis(self.mapping, X, j)
            w += interp[self.element_dofs[j, :]][:, None]*basis[0]

        nt = self.nelems
        t = np.tile(m.t, (1, nt))
        dt = np.max(t)
        t += (dt + 1) * np.tile(np.arange(nt), (m.t.shape[0]*m.t.shape[1], 1)).flatten('F').reshape((-1, m.t.shape[0])).T

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

        finder = self.mesh.element_finder()

        def interpfun(x):
            tris = finder(*x)
            pts = self.mapping.invF(x[:, :, np.newaxis], tind=tris)
            w = np.zeros(x.shape[1])
            for k in range(self.Nbfun):
                phi = self.elem.gbasis(self.mapping, pts, k, tind=tris)
                w += y[self.element_dofs[k, tris]] * phi[0].flatten()
            return w

        return interpfun
