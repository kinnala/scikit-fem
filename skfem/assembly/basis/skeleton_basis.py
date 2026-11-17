from typing import List, Optional, Tuple, Any

import numpy as np
from numpy import ndarray
from skfem.element import Element
from skfem.generic_utils import OrientedBoundary

from .composite_basis import CompositeBasis
from .cell_basis import CellBasis
from .facet_basis import FacetBasis


class SkeletonBasis(CompositeBasis):
    """A pair of bases for hybridized methods.

    The first component is the trace of a function defined in the elements
    and the second component is a function defined on the facets of the mesh
    (the mesh skeleton).  The pair is evaluated over all element-facet pairs,
    i.e. over the facets of each element, so that an interior facet is
    included twice, once from each side.  Consequently, the normal vector
    ``w.n`` is the outward normal of the element.  The DOFs are numbered so
    that the element DOFs come first followed by the skeleton DOFs.

    Usually initialized using :meth:`~skfem.assembly.CellBasis.skeleton`,
    e.g., for a hybridizable discontinuous Galerkin method:

    >>> import numpy as np
    >>> from scipy.sparse import bmat, csr_matrix
    >>> from skfem import (MeshTri, Basis, BilinearForm, LinearForm,
    ...                    ElementTriP1DG, ElementLineP1DG, solve, condense)
    >>> from skfem.helpers import dot, grad
    >>> m = MeshTri().refined(3)
    >>> basis = Basis(m, ElementTriP1DG())
    >>> sbasis = basis.skeleton(ElementLineP1DG())
    >>> @BilinearForm
    ... def hdg(u, ut, v, vt, w):
    ...     return (dot(w.n, grad(u)) * (vt - v) + dot(w.n, grad(v)) * (ut - u)
    ...             + 10. / w.h * (ut - u) * (vt - v))
    >>> laplace = BilinearForm(lambda u, v, w: dot(grad(u), grad(v)))
    >>> nt = sbasis.facets.N
    >>> K = bmat([[laplace.assemble(basis), None],
    ...           [None, csr_matrix((nt, nt))]]) + hdg.assemble(sbasis)
    >>> f = np.concatenate((LinearForm(lambda v, w: v).assemble(basis),
    ...                     np.zeros(nt)))
    >>> x = solve(*condense(K, f, D=basis.N + sbasis.get_dofs()))
    >>> (u, ubasis), (ut, utbasis) = sbasis.split(x)
    >>> print(round(float(u.max()), 4))
    0.0749

    Note that the zero block is needed because :func:`scipy.sparse.bmat`
    drops a block row and column without any matrices.

    The element DOFs can be eliminated element-by-element (hybridization)
    since the element block is block diagonal:

    >>> (Ah, B), (C, D) = hdg.elemental_blocks(sbasis)
    >>> lap = laplace.elemental(basis)
    >>> Ainv = lap.fromlocal(lap.tolocal() + Ah.tolocal(sbasis)).inverse()
    >>> Ainv = Ainv.tocsr()
    >>> B, C, D = B.tocsr(), C.tocsr(), D.tocsr()
    >>> f = LinearForm(lambda v, w: v).assemble(basis)
    >>> ut = solve(*condense(D - C @ Ainv @ B, -C @ (Ainv @ f),
    ...                      D=sbasis.get_dofs()))
    >>> u = Ainv @ (f - B @ ut)
    >>> print(round(float(u.max()), 4))
    0.0749

    """

    def __init__(self,
                 basis: CellBasis,
                 elem: Element,
                 intorder: Optional[int] = None,
                 quadrature: Optional[Tuple[ndarray, ndarray]] = None):
        """Initialize the pair of bases.

        Parameters
        ----------
        basis
            The basis for the function defined in the elements.
        elem
            The element for the function defined on the skeleton, e.g.,
            :class:`~skfem.element.ElementLineP1DG` for triangular meshes.
            Continuity across the vertices of the skeleton is defined by
            the element, e.g., discontinuous elements give hybridizable
            discontinuous Galerkin methods.
        intorder
            Optional integration order.  By default, based on the element of
            ``basis``.
        quadrature
            Optional tuple of quadrature points and weights.

        """
        mesh = basis.mesh
        if basis.tind is not None:
            raise NotImplementedError("Skeleton of a subdomain not "
                                      "supported.")
        if mesh.dim() != mesh.gdim() or mesh.dim() == 1:
            raise NotImplementedError("Skeleton supported only for two- and "
                                      "three-dimensional meshes.")
        if mesh.doflocs.shape[1] != mesh.nvertices:
            # the skeleton mesh would consist of straight facets
            raise NotImplementedError("Skeleton not supported for high "
                                      "order meshes.")
        mtype, _ = mesh._p0(mesh.brefdom)
        if elem.refdom != mtype.elem.refdom:
            raise ValueError("The element of the skeleton must be defined "
                             "on {}.".format(mesh.brefdom.__name__))

        skeleton, _ = mesh.trace(np.arange(mesh.nfacets, dtype=np.int32),
                                 mtype=mtype)

        # each element-facet pair once: side 0 of all facets and side 1 of
        # the interior facets
        interior = np.nonzero(mesh.f2t[1] >= 0)[0].astype(np.int32)
        pairs = OrientedBoundary(
            np.concatenate((np.arange(mesh.nfacets, dtype=np.int32),
                            interior)),
            np.concatenate((np.zeros(mesh.nfacets, dtype=np.int64),
                            np.ones(len(interior), dtype=np.int64))),
        )
        fbasis = FacetBasis(mesh,
                            basis.elem,
                            mapping=basis.mapping,
                            intorder=intorder,
                            quadrature=quadrature,
                            facets=pairs,
                            dofs=basis.dofs)
        # the element k of 'skeleton' is the facet k of 'mesh'
        tbasis = CellBasis(skeleton,
                           elem,
                           quadrature=fbasis.quadrature,
                           elements=np.asarray(pairs))
        super().__init__(fbasis, tbasis)

        #: The basis for the function defined in the elements.
        self.volume = basis
        #: The mesh of the function defined in the elements.
        self.mesh = mesh
        #: The element of each element-facet pair, e.g., for
        #: :meth:`~skfem.assembly.form.COOData.tolocal`.
        self.tind = fbasis.tind
        #: The basis for the function defined on the skeleton with one
        #: element per facet.
        self.facets = CellBasis(skeleton, elem)

    def split(self, x: ndarray) -> List[Tuple[ndarray, Any]]:
        """Split a solution vector into the element and skeleton parts.

        Returns ``[(u, self.volume), (ut, self.facets)]``.

        """
        n = self.volume.N
        return [(x[:n], self.volume), (x[n:], self.facets)]

    def get_dofs(self,
                 facets: Optional[Any] = None,
                 elements: Optional[Any] = None,
                 nodes: Optional[Any] = None,
                 skip: Optional[List[str]] = None,
                 edges: Optional[Any] = None) -> ndarray:
        """Return the skeleton DOFs on the given facets.

        Parameters
        ----------
        facets
            Anything that can be passed to
            :meth:`~skfem.mesh.Mesh.normalize_facets`, e.g., the name of a
            boundary.  By default, the boundary facets.  The other
            parameters of :meth:`~skfem.assembly.CellBasis.get_dofs` are not
            supported; use ``self.volume`` or ``self.facets``.

        Returns
        -------
        ndarray
            The DOFs in the numbering of the skeleton unknown, i.e. as in
            ``self.facets``.  Add ``self.volume.N`` to obtain the DOFs in the
            numbering of the pair.

        """
        if any(arg is not None for arg in (elements, nodes, skip, edges)):
            raise NotImplementedError("Only facets are supported.")
        facets = np.asarray(self.volume.mesh.normalize_facets(facets))
        return self.facets.get_dofs(elements=facets).flatten()

    def __repr__(self):
        return ("<skfem SkeletonBasis({}, {}) object>\n"
                "  Number of element DOFs: {}\n"
                "  Number of skeleton DOFs: {}"
                .format(type(self.volume.elem).__name__,
                        type(self.facets.elem).__name__,
                        self.volume.N,
                        self.facets.N))
