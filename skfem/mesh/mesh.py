from dataclasses import dataclass, replace
from typing import Callable, Dict, List, Optional, Tuple, Type, Union
from warnings import warn

import numpy as np
from numpy import ndarray

from ..element import BOUNDARY_ELEMENT_MAP, Element


@dataclass(repr=False)
class Mesh:
    """A mesh defined by :class:`~skfem.element.Element` class.

    :class:`~skfem.mesh.Mesh` is defined as a combination of elements/cells by
    specifying the spatial locations of the finite element nodes.

    """

    doflocs: ndarray  #: The locations of the finite element nodes
    t: ndarray  #: The connectivity of the elements/cells
    _boundaries: Optional[Dict[str, ndarray]] = None
    _subdomains: Optional[Dict[str, ndarray]] = None
    elem: Type[Element] = Element
    affine: bool = False  #: Use :class:`~skfem.mapping.MappingAffine`?
    # Some parts of the library, most notably the normal vector construction in
    # ElementGlobal._eval_dofs, assume that the element indices are ascending
    # because this leads to consistent normal vectors for both elements sharing
    # a facet.  Therefore, the element indices are sorted in a triangle mesh.
    # However, some algorithms (e.g., adaptive refinement) require switching
    # off this behaviour and, hence, this flag exists.
    sort_t: bool = False

    @property
    def p(self):
        return self.doflocs

    @property
    def dofs(self):
        from skfem.assembly import Dofs
        if not hasattr(self, '_dofs'):
            self._dofs = Dofs(self, self.elem())
        return self._dofs

    @property
    def refdom(self):
        return self.elem.refdom

    @property
    def brefdom(self):
        return self.elem.refdom.brefdom

    @property
    def bndelem(self):
        if self.elem in BOUNDARY_ELEMENT_MAP:
            return BOUNDARY_ELEMENT_MAP[self.elem]()
        return None

    @property
    def nelements(self):
        return self.t.shape[1]

    @property
    def nvertices(self):
        return np.max(self.t) + 1

    @property
    def nfacets(self):
        return self.facets.shape[1]

    @property
    def nedges(self):
        return self.edges.shape[1]

    @property
    def nnodes(self):
        return self.t.shape[0]

    @property
    def subdomains(self):
        return self._subdomains

    @property
    def boundaries(self):
        return self._boundaries

    @property
    def facets(self):
        if not hasattr(self, '_facets'):
            self._init_facets()
        return self._facets

    @property
    def t2f(self):
        if not hasattr(self, '_t2f'):
            self._init_facets()
        return self._t2f

    @property
    def f2t(self):
        if not hasattr(self, '_f2t'):
            self._f2t = self.build_inverse(self.t, self.t2f)
        return self._f2t

    @property
    def edges(self):
        if not hasattr(self, '_edges'):
            self._init_edges()
        return self._edges

    @property
    def t2e(self):
        if not hasattr(self, '_t2e'):
            self._init_edges()
        return self._t2e

    def dim(self):
        return self.elem.refdom.dim()

    def boundary_facets(self) -> ndarray:
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1] == -1)[0]

    def boundary_edges(self) -> ndarray:
        """Return an array of boundary edge indices."""
        facets = self.boundary_facets()
        boundary_edges = np.sort(np.hstack(
            tuple([np.vstack((self.facets[itr, facets],
                              self.facets[(itr + 1) % self.facets.shape[0],
                              facets]))
                   for itr in range(self.facets.shape[0])])).T, axis=1)
        edge_candidates = np.unique(self.t2e[:, self.f2t[0, facets]])
        A = self.edges[:, edge_candidates].T
        B = boundary_edges
        dims = A.max(0) + 1
        ix = np.where(np.in1d(
            np.ravel_multi_index(A.T, dims),  # type: ignore
            np.ravel_multi_index(B.T, dims),  # type: ignore
        ))[0]
        return edge_candidates[ix]

    def with_boundaries(self,
                        boundaries: Dict[str, Callable[[ndarray], ndarray]],
                        boundaries_only: bool = True):
        """Return a copy of the mesh with named boundaries.

        Parameters
        ----------
        boundaries
            A dictionary of lambda functions with the names of the boundaries
            as keys.  The midpoint of the facet should return ``True`` for the
            corresponding lambda function if the facet belongs to the boundary.
        boundaries_only
            If ``True``, consider only facets on the boundary of the domain.

        """
        return replace(
            self,
            _boundaries={
                **({} if self._boundaries is None else self._boundaries),
                **{name: self.facets_satisfying(test, boundaries_only)
                   for name, test in boundaries.items()}
            },
        )

    def with_subdomains(self,
                        subdomains: Dict[str, Callable[[ndarray], ndarray]]):
        """Return a copy of the mesh with named subdomains.

        Parameters
        ----------
        boundaries
            A dictionary of lambda functions with the names of the subdomains
            as keys.  The midpoint of the element should return ``True`` for
            the corresponding lambda function if the element belongs to the
            subdomain.

        """
        return replace(
            self,
            _subdomains={
                **({} if self._subdomains is None else self._subdomains),
                **{name: self.elements_satisfying(test)
                   for name, test in subdomains.items()},
            }
        )

    def _encode_point_data(self) -> Dict[str, List[ndarray]]:

        subdomains = {} if self._subdomains is None else self._subdomains
        boundaries = {} if self._boundaries is None else self._boundaries

        def indicator(ix):
            ind = np.zeros(self.nvertices)
            ind[ix] = 1
            return ind

        return {
            **{
                f"skfem:s:{name}": indicator(np.unique(self.t[:, subdomain]
                                                       .flatten()))
                for name, subdomain in subdomains.items()
            },
            **{
                f"skfem:b:{name}": indicator(np.unique(self.facets[:, boundary]
                                                       .flatten()))
                for name, boundary in boundaries.items()
            },
        }

    def _encode_cell_data(self) -> Dict[str, List[ndarray]]:

        subdomains = {} if self._subdomains is None else self._subdomains
        boundaries = {} if self._boundaries is None else self._boundaries

        return {
            **{
                f"skfem:s:{name}": [
                    np.isin(np.arange(self.t.shape[1]), subdomain).astype(int)
                ]
                for name, subdomain in subdomains.items()
            },
            **{
                f"skfem:b:{name}": [
                    ((1 << np.arange(self.t2f.shape[0]))
                     @ np.isin(self.t2f, boundary)).astype(int)
                ]
                for name, boundary in boundaries.items()
            },
        }

    def _decode_cell_data(self, cell_data: Dict[str, List[ndarray]]):

        subdomains = {}
        boundaries = {}

        for name, data in cell_data.items():
            subnames = name.split(":")
            if subnames[0] != "skfem":
                continue
            if subnames[1] == "s":
                subdomains[subnames[2]] = np.nonzero(data[0])[0]
            elif subnames[1] == "b":
                boundaries[subnames[2]] = self.t2f[
                    (1 << np.arange(self.t2f.shape[0]))[:, None]
                    & data[0].astype(np.int64) > 0
                ]

        return boundaries, subdomains

    def boundary_nodes(self) -> ndarray:
        """Return an array of boundary node indices."""
        return np.unique(self.facets[:, self.boundary_facets()])

    def interior_nodes(self) -> ndarray:
        """Return an array of interior node indices."""
        return np.setdiff1d(np.arange(0, self.p.shape[1]),
                            self.boundary_nodes())

    def nodes_satisfying(self,
                         test: Callable[[ndarray], ndarray],
                         boundaries_only: bool = False) -> ndarray:
        """Return nodes that satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the set of nodes that are to
            be included in the return set.
        boundaries_only
            If ``True``, include only boundary facets.

        """
        nodes = np.nonzero(test(self.p))[0]
        if boundaries_only:
            nodes = np.intersect1d(nodes, self.boundary_nodes())
        return nodes

    def facets_satisfying(self,
                          test: Callable[[ndarray], ndarray],
                          boundaries_only: bool = False) -> ndarray:
        """Return facets whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the facet midpoints that are
            to be included in the return set.
        boundaries_only
            If ``True``, include only boundary facets.

        """
        midp = [np.sum(self.p[itr, self.facets], axis=0) / self.facets.shape[0]
                for itr in range(self.dim())]
        facets = np.nonzero(test(np.array(midp)))[0]
        if boundaries_only:
            facets = np.intersect1d(facets, self.boundary_facets())
        return facets

    def elements_satisfying(self,
                            test: Callable[[ndarray], ndarray]) -> ndarray:
        """Return elements whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the element midpoints that
            are to be included in the return set.

        """
        midp = [np.sum(self.p[itr, self.t], axis=0) / self.t.shape[0]
                for itr in range(self.dim())]
        return np.nonzero(test(np.array(midp)))[0]

    def _expand_facets(self, ix: ndarray) -> Tuple[ndarray, ndarray]:
        """Return vertices and edges corresponding to given facet indices.

        Parameters
        ----------
        ix
            An array of facet indices.

        """
        vertices = np.unique(self.facets[:, ix].flatten())

        if self.dim() == 3:
            edge_candidates = self.t2e[:, self.f2t[0, ix]].flatten()
            # subset of edges that share all points with the given facets
            subset = np.nonzero(
                np.prod(np.isin(self.edges[:, edge_candidates],
                                self.facets[:, ix].flatten()),
                        axis=0)
            )[0]
            edges = np.intersect1d(self.boundary_edges(),
                                   edge_candidates[subset])
        else:
            edges = np.array([], dtype=np.int64)

        return vertices, edges

    def _mapping(self):
        """Return a default reference mapping for the mesh."""
        from skfem.mapping import MappingAffine, MappingIsoparametric
        if not hasattr(self, '_cached_mapping'):
            if self.affine:
                self._cached_mapping = MappingAffine(self)
            else:
                self._cached_mapping = MappingIsoparametric(
                    self,
                    self.elem(),
                    self.bndelem,
                )
        return self._cached_mapping

    def _init_facets(self):
        """Initialize ``self.facets``."""
        self._facets, self._t2f = self.build_entities(
            self.t,
            self.elem.refdom.facets,
        )

    def _init_edges(self):
        """Initialize ``self.edges``."""
        self._edges, self._t2e = self.build_entities(
            self.t,
            self.elem.refdom.edges,
        )

    def __post_init__(self):
        """Support node orders used in external formats.

        We expect ``self.doflocs`` to be ordered based on the
        degrees-of-freedom in :class:`skfem.assembly.Dofs`.  External formats
        for high order meshes commonly use a less strict ordering scheme and
        the extra nodes are described as additional rows in ``self.t``.  This
        method attempts to accommodate external formas by reordering
        ``self.doflocs`` and changing the indices in ``self.t``.

        """
        if self.sort_t:
            self.t = np.sort(self.t, axis=0)

        self.doflocs = np.asarray(self.doflocs, dtype=np.float64, order="K")
        self.t = np.asarray(self.t, dtype=np.int64, order="K")

        M = self.elem.refdom.nnodes

        if self.nnodes > M:
            # reorder DOFs to the expected format: vertex DOFs are first
            p, t = self.doflocs, self.t
            t_nodes = t[:M]
            uniq, ix = np.unique(t_nodes, return_inverse=True)
            self.t = (np.arange(len(uniq), dtype=np.int64)[ix]
                      .reshape(t_nodes.shape))
            doflocs = np.hstack((
                p[:, uniq],
                np.zeros((p.shape[0], np.max(t) + 1 - len(uniq))),
            ))
            doflocs[:, self.dofs.element_dofs[M:].flatten('F')] =\
                p[:, t[M:].flatten('F')]
            self.doflocs = doflocs

        # C_CONTIGUOUS is more performant in dimension-based slices
        if not self.doflocs.flags['C_CONTIGUOUS']:
            if self.doflocs.shape[1] > 1000:
                warn("Transforming over 1000 vertices to C_CONTIGUOUS.")
            self.doflocs = np.ascontiguousarray(self.doflocs)

        if not self.t.flags['C_CONTIGUOUS']:
            if self.t.shape[1] > 1000:
                warn("Transforming over 1000 elements to C_CONTIGUOUS.")
            self.t = np.ascontiguousarray(self.t)

    def __rmatmul__(self, other):
        out = self.__matmul__(other)
        return out[1:] + out[0:1]

    def __matmul__(self, other):
        """Join meshes with different types; return a list of meshes."""
        cls = type(self)
        if isinstance(other, Mesh):
            other = [other]
        if isinstance(other, list):
            p = np.hstack((self.p,) + tuple([mesh.p for mesh in other]))
            pT = np.ascontiguousarray(p.T)
            _, ixa, ixb = np.unique(pT.view([('', pT.dtype)] * pT.shape[1]),
                                    return_index=True, return_inverse=True)
            p = p[:, ixa]
            return [
                cls(p, ixb[self.t]),
                *[type(m)(p, ixb[m.t + self.p.shape[1]])
                  for i, m in enumerate(other)],
            ]
        raise NotImplementedError

    def is_valid(self) -> bool:
        """Perform some mesh validation checks."""
        # check that there are no duplicate points
        tmp = np.ascontiguousarray(self.p.T)
        if self.p.shape[1] != np.unique(tmp.view([('', tmp.dtype)]
                                                 * tmp.shape[1])).shape[0]:
            warn("Mesh contains duplicate vertices.")
            return False

        # check that all points are at least in some element
        if len(np.setdiff1d(np.arange(self.p.shape[1]),
                            np.unique(self.t))) > 0:
            warn("Mesh contains a vertex not belonging to any element.")
            return False

        return True

    def __add__(self, other):
        """Join two meshes."""
        cls = type(self)
        if not isinstance(other, cls):
            raise TypeError("Can only join meshes with same type.")
        p = np.hstack((self.p.round(decimals=8),
                       other.p.round(decimals=8)))
        t = np.hstack((self.t, other.t + self.p.shape[1]))
        tmp = np.ascontiguousarray(p.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        p = p[:, ixa]
        t = ixb[t]
        return cls(p, t)

    def __repr__(self):
        rep = ""
        rep += "<skfem {} object>\n".format(type(self).__name__)
        rep += "  Number of elements: {}\n".format(self.nelements)
        rep += "  Number of vertices: {}\n".format(self.nvertices)
        rep += "  Number of nodes: {}".format(self.p.shape[1])
        if self.subdomains is not None:
            rep += "\n  Named subdomains [# elements]: {}".format(
                ', '.join(
                    map(lambda k: '{} [{}]'.format(k, len(self.subdomains[k])),
                        list(self.subdomains.keys()))
                )
            )
        if self.boundaries is not None:
            rep += "\n  Named boundaries [# facets]: {}".format(
                ', '.join(
                    map(lambda k: '{} [{}]'.format(k, len(self.boundaries[k])),
                        list(self.boundaries.keys()))
                )
            )
        return rep

    def __str__(self):
        return self.__repr__()

    def save(self,
             filename: str,
             point_data: Optional[Dict[str, ndarray]] = None,
             cell_data: Optional[Dict[str, ndarray]] = None,
             **kwargs) -> None:
        """Export the mesh and fields using meshio.

        Parameters
        ----------
        filename
            The output filename, with suffix determining format;
            e.g. .msh, .vtk, .xdmf
        point_data
            Data related to the vertices of the mesh.
        cell_data
            Data related to the elements of the mesh.

        """
        from skfem.io.meshio import to_file
        return to_file(self,
                       filename,
                       point_data,
                       cell_data,
                       **kwargs)

    @classmethod
    def load(cls,
             filename: str,
             out: Optional[List[str]] = None,
             **kwargs):
        """Load a mesh using meshio.

        Parameters
        ----------
        filename
            The filename of the mesh file.
        out
            Optional list of ``meshio.Mesh`` attribute names, overwrite with
            the corresponding data.  E.g., ``['point_data', 'cell_data']``.

        """
        from skfem.io.meshio import from_file
        return from_file(filename,
                         out,
                         **kwargs)

    @classmethod
    def from_dict(cls, d):
        from skfem.io.json import from_dict
        return from_dict(cls, d)

    def to_dict(self):
        from skfem.io.json import to_dict
        return to_dict(self)

    @classmethod
    def from_mesh(cls, mesh, t: Optional[ndarray] = None):
        """Reuse an existing mesh by adding the higher order nodes.

        Parameters
        ----------
        mesh
            The mesh used in the initialization.  Connectivity of the new mesh
            will match ``mesh.t`` unless ``t`` is given
        t
            Optionally specify new connectivity for the resulting mesh.

        """
        from skfem.assembly import Dofs

        mapping = mesh._mapping()
        nelem = cls.elem
        dofs = Dofs(mesh, nelem())
        locs = mapping.F(nelem.doflocs.T)
        doflocs = np.zeros((locs.shape[0], dofs.N))

        assert dofs.element_dofs is not None

        # match mapped dofs and global dof numbering
        for itr in range(locs.shape[0]):
            for jtr in range(dofs.element_dofs.shape[0]):
                doflocs[itr, dofs.element_dofs[jtr]] = locs[itr, :, jtr]

        return cls(
            doflocs=doflocs,
            t=t if t is not None else mesh.t,
        )

    @classmethod
    def init_refdom(cls):
        """Initialize a mesh corresponding to the reference domain."""
        return cls(cls.elem.refdom.p, cls.elem.refdom.t)

    def refined(self, times_or_ix: Union[int, ndarray] = 1):
        """Return a refined mesh.

        Parameters
        ----------
        times_or_ix
            Either an integer giving the number of uniform refinements or an
            array of element indices for adaptive refinement.

        """
        m = self
        if isinstance(times_or_ix, int):
            for _ in range(times_or_ix):
                m = m._uniform()
        else:
            m = m._adaptive(times_or_ix)
        return m

    def scaled(self, factors):
        """Return a new mesh with scaled dimensions.

        Parameters
        ----------
        factors
            Scale each dimension by a factor.

        """
        if isinstance(factors, float):
            # for backwards compatibility
            factors = self.doflocs.shape[0] * [factors]
        return replace(
            self,
            doflocs=np.array([self.doflocs[itr] * factors[itr]
                              for itr in range(len(factors))]),
        )

    def translated(self, diffs):
        """Return a new translated mesh.

        Parameters
        ----------
        diffs
            Translate the mesh by a vector. Must have same size as the mesh
            dimension.

        """
        return replace(
            self,
            doflocs=np.array([self.doflocs[itr] + diffs[itr]
                              for itr in range(len(diffs))]),
        )

    def mirrored(self,
                 normal: Tuple[float, ...],
                 point: Optional[Tuple[float, ...]] = None):
        """Return a mesh mirrored with respect to a normal.

        Meant to be combined with the other methods to build more general
        meshes, e.g.,

        >>> from skfem import MeshTet
        >>> m1 = MeshTet()
        >>> m2 = m1.mirrored((1, 0, 0))
        >>> m3 = m1.mirrored((0, 1, 0))
        >>> m4 = m1.mirrored((0, 0, 1))
        >>> m = m1 + m2 + m3 + m4
        >>> (m.nvertices, m.nelements)
        (20, 20)

        Parameters
        ----------
        normal
            The normal vector of the mirror plane.
        point
            An optional point through which the plane passes. By default, the
            point corresponds to the origin.

        """
        if point is None:
            point = (0,) * self.dim()

        p = self.p.copy()
        p0 = np.array(point)
        n = np.array(normal)
        n = n / np.linalg.norm(n)
        p = p - 2. * np.dot(n, p - p0[:, None]) * n[:, None]

        return replace(
            self,
            doflocs=p,
        )

    def _uniform(self):
        """Perform a single uniform refinement."""
        raise NotImplementedError

    def _adaptive(self, ix: ndarray):
        """Adaptively refine the given set of elements."""
        raise NotImplementedError

    def _splitref(self, nrefs: int = 1):
        """Split mesh into separate nonconnected elements and refine.

        Used for visualization purposes.

        Parameters
        ----------
        nrefs
            The number of refinements.

        """
        cls = type(self)
        m = cls.init_refdom().refined(nrefs)
        X = m.p
        x = self._mapping().F(m.p)

        # create connectivity for the new mesh
        nt = self.nelements
        t = np.tile(m.t, (1, nt))
        dt = np.max(t)
        t += ((dt + 1)
              * (np.tile(np.arange(nt), (m.t.shape[0] * m.t.shape[1], 1))
                 .flatten('F')
                 .reshape((-1, m.t.shape[0])).T))

        if X.shape[0] == 1:
            p = np.array([x.flatten()])
        else:
            p = x[0].flatten()
            for itr in range(len(x) - 1):
                p = np.vstack((p, x[itr + 1].flatten()))

        return cls(p, t)

    @staticmethod
    def build_entities(t, indices, sort=True):
        """Build low dimensional topological entities."""
        if indices is None:
            return None, None
        indexing = np.hstack(tuple([t[ix] for ix in indices]))
        sorted_indexing = np.sort(indexing, axis=0)

        sorted_indexing, ixa, ixb = np.unique(sorted_indexing,
                                              axis=1,
                                              return_index=True,
                                              return_inverse=True)
        mapping = ixb.reshape((len(indices), t.shape[1]))

        if sort:
            return np.ascontiguousarray(sorted_indexing), mapping

        return np.ascontiguousarray(indexing[:, ixa]), mapping

    @staticmethod
    def build_inverse(t, mapping):
        """Build inverse mapping from low dimensional topological entities."""
        e = mapping.flatten(order='C')
        tix = np.tile(np.arange(t.shape[1]), (1, t.shape[0]))[0]

        e_first, ix_first = np.unique(e, return_index=True)
        e_last, ix_last = np.unique(e[::-1], return_index=True)
        ix_last = e.shape[0] - ix_last - 1

        inverse = np.zeros((2, np.max(mapping) + 1), dtype=np.int64)
        inverse[0, e_first] = tix[ix_first]
        inverse[1, e_last] = tix[ix_last]

        inverse[1, np.nonzero(inverse[0] == inverse[1])[0]] = -1

        return inverse

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        """Fallback for 3D meshes."""
        return p

    def param(self) -> float:
        """Return mesh parameter, viz the length of the longest edge."""
        raise NotImplementedError

    def _reix(self, ix: ndarray) -> Tuple[ndarray, ndarray]:
        """Connect ``self.p`` based on the indices ``ix``."""
        ixuniq = np.unique(ix)
        t = np.zeros(np.max(ix) + 1, dtype=np.int64)
        t[ixuniq] = np.arange(len(ixuniq), dtype=np.int64)
        return self.p[:, ixuniq], t[ix]

    def remove_elements(self, element_indices: ndarray):
        """Construct a new mesh by removing elements.

        Parameters
        ----------
        element_indices
            List of element indices to remove.

        """
        p, t = self._reix(np.delete(self.t, element_indices, axis=1))
        return replace(
            self,
            doflocs=p,
            t=t,
        )

    def element_finder(self, mapping=None):
        """Return a function handle from location to element index.

        Parameters
        ----------
        mapping
            The affine mapping for the mesh.

        """
        raise NotImplementedError

    def draw(self, *args, **kwargs):
        """Convenience wrapper for vedo."""
        from skfem.visuals.vedo import draw
        return draw(self, *args, **kwargs)
