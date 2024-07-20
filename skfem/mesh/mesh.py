import logging
import importlib

from dataclasses import dataclass, replace
from os import PathLike
from typing import Callable, Dict, List, Optional, Tuple, Type, Union

import numpy as np
from numpy import ndarray

from ..element import BOUNDARY_ELEMENT_MAP, Element
from ..generic_utils import OrientedBoundary


logger = logging.getLogger(__name__)


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
    validate: bool = True  # run validation check if log_level<=DEBUG

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
        if self.edges is not None:
            return self.edges.shape[1]
        raise NotImplementedError

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
    def f2e(self):
        if not hasattr(self, '_f2e'):
            _, self._f2e = self.build_entities(
                self.facets,
                self.bndelem.refdom.facets,
            )
        return self._f2e

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

    @property
    def p2f(self):
        """Incidence matrix between facets and vertices.

        Examples
        --------
        To find facets with point 5:

        >>> import numpy as np
        >>> from skfem import MeshTet
        >>> mesh = MeshTet().refined()
        >>> np.nonzero(mesh.p2f[:, 5])[0]
        array([33, 34, 35], dtype=int32)

        """
        from scipy.sparse import coo_matrix
        facets = self.facets.flatten('C')
        return coo_matrix(
            (np.ones(len(facets), dtype=np.int32),
             (np.concatenate((np.arange(self.nfacets),)
                             * self.facets.shape[0]), facets)),
            shape=(self.nfacets, self.nvertices),
            dtype=np.int32,
        ).tocsc()

    @property
    def p2t(self):
        """Incidence matrix between elements and vertices."""
        from scipy.sparse import coo_matrix
        t = self.t.flatten('C')
        return coo_matrix(
            (np.ones(len(t), dtype=np.int32),
             (np.concatenate((np.arange(self.nelements),)
                             * self.nnodes), t)),
            shape=(self.nelements, self.nvertices),
            dtype=np.int32,
        ).tocsc()

    @property
    def p2e(self):
        """Incidence matrix between edges and vertices."""
        from scipy.sparse import coo_matrix
        edges = self.edges.flatten('C')
        return coo_matrix(
            (np.ones(len(edges), dtype=np.int32),
             (np.concatenate((np.arange(self.nedges),)
                             * self.edges.shape[0]), edges)),
            shape=(self.nedges, self.nvertices),
            dtype=np.int32,
        ).tocsc()

    @property
    def e2t(self):
        """Incidence matrix between elements and edges."""
        p2t = self.p2t
        edges = self.edges
        return p2t[:, edges[0]].multiply(p2t[:, edges[1]])

    def dim(self):
        return self.elem.refdom.dim()

    def boundary_facets(self) -> ndarray:
        """Return an array of boundary facet indices."""
        return np.nonzero(self.f2t[1] == -1)[0].astype(np.int32)

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
        ix = np.where(np.isin(
            np.ravel_multi_index(A.T, dims),  # type: ignore
            np.ravel_multi_index(B.T, dims),  # type: ignore
        ))[0]
        return edge_candidates[ix]

    def with_defaults(self):
        """Return a copy with the default tags ('left', 'right', ...)."""
        return self.with_boundaries(self._build_default_tags())

    def with_boundaries(self,
                        boundaries: Dict[str, Union[Callable[[ndarray],
                                                             ndarray],
                                                    ndarray]],
                        boundaries_only: bool = True):
        """Return a copy of the mesh with named boundaries.

        Parameters
        ----------
        boundaries
            A dictionary of index arrays or lambda functions with the names of
            the boundaries as keys.  The midpoint of the facet should return
            ``True`` for the corresponding lambda function if the facet belongs
            to the boundary.
        boundaries_only
            If ``True``, consider only facets on the boundary of the domain.

        """
        return replace(
            self,
            _boundaries={
                **({} if self._boundaries is None else self._boundaries),
                **{name: self.facets_satisfying(test_or_set, boundaries_only)
                   if callable(test_or_set) else test_or_set
                   for name, test_or_set in boundaries.items()}
            },
        )

    def with_subdomains(self,
                        subdomains: Dict[str, Union[Callable[[ndarray],
                                                             ndarray],
                                                    ndarray]]):
        """Return a copy of the mesh with named subdomains.

        Parameters
        ----------
        subdomains
            A dictionary of lambda functions with the names of the subdomains
            as keys.  The midpoint of the element should return ``True`` for
            the corresponding lambda function if the element belongs to the
            subdomain.

        """
        return replace(
            self,
            _subdomains={
                **({} if self._subdomains is None else self._subdomains),
                **{name: (self.elements_satisfying(test)
                          if callable(test) else test)
                   for name, test in subdomains.items()},
            },
        )

    def _build_default_tags(self):

        boundaries = {}
        # default boundary names along the dimensions
        minnames = ['left', 'bottom', 'front']
        maxnames = ['right', 'top', 'back']
        for d in range(self.doflocs.shape[0]):
            dmin = np.min(self.doflocs[d])
            ix = self.facets_satisfying(lambda x: x[d] == dmin)
            if len(ix) >= 1:
                boundaries[minnames[d]] = ix
        for d in range(self.doflocs.shape[0]):
            dmax = np.max(self.doflocs[d])
            ix = self.facets_satisfying(lambda x: x[d] == dmax)
            if len(ix) >= 1:
                boundaries[maxnames[d]] = ix

        return boundaries

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

        def encode_boundary(boundary: Union[ndarray, OrientedBoundary]
                            ) -> ndarray:
            """return an array with an int per cell encoding a 'boundary'

            i.e. a subset of (optionally oriented) facets.

            Although these are stored internally as arrays of indices of
            facets, with an optional array of 0/1 orientations, an alternative
            representation is as a boolean array of shape (self.nelements,
            self.refdom.nfacets) with entries `True` if that facet belongs to
            the boundary.  This array can be bit-packed to give a one-
            dimensional array of ints, one per cell.  This is convenient as it
            can be treated as `cell_data` by `skfem.io.meshio` and stored in
            many external formats (VTK, XDMF, ...).

            That is, the binary expansion of the int for a cell gives the bit-
            flags for whether the corresponding facets are in `boundary`.  This
            naturally implies the orientation, as the cell is on the inside of
            the facet.
            """
            b = (
                boundary
                if isinstance(boundary, OrientedBoundary)
                else OrientedBoundary(boundary, np.zeros_like(boundary))
            )
            t2f_mask = np.zeros_like(self.t2f)
            columns = self.f2t[(b.ori, b)]
            r, c = np.nonzero(self.t2f[:, columns] == b)
            t2f_mask[(r, columns[c])] = 1
            return (1 << np.arange(self.refdom.nfacets)) @ t2f_mask

        return {
            **{
                f"skfem:s:{name}": [
                    np.isin(np.arange(self.t.shape[1],
                                      dtype=np.int32), subdomain).astype(int)
                ]
                for name, subdomain in subdomains.items()
            },
            **{
                f"skfem:b:{name}": [encode_boundary(boundary)]
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
                mask = (
                    (1 << np.arange(self.refdom.nfacets))[:, None]
                    & data[0].astype(np.int32)
                ).astype(bool)
                facets = np.sort(self.t2f[mask])
                cells = mask.nonzero()[1]
                ori = np.arange(2) @ (self.f2t[:, facets] == cells)
                boundaries[subnames[2]] = (
                    OrientedBoundary(facets, ori) if ori.any() else facets
                )

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
        nodes = np.nonzero(test(self.p))[0].astype(np.int32)
        if boundaries_only:
            nodes = np.intersect1d(nodes, self.boundary_nodes())
        return nodes

    def facets_satisfying(self,
                          test: Callable[[ndarray], ndarray],
                          boundaries_only: bool = False,
                          normal: Optional[ndarray] = None) -> ndarray:
        """Return facets whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the facet midpoints that are
            to be included in the return set.
        boundaries_only
            If ``True``, include only boundary facets.
        normal
            If given, used to orient the set of facets.

        """
        midp = self.p[:, self.facets].mean(axis=1)
        facets = np.nonzero(test(midp))[0].astype(np.int32)
        if boundaries_only:
            facets = np.intersect1d(facets, self.boundary_facets())
        if normal is not None:
            tind = self.f2t[0, facets]
            mapping = self._mapping()
            normals = mapping.normals(np.zeros((self.dim(), 1)),
                                      tind,
                                      facets,
                                      self.t2f).T[0].T
            ori = 1 * (np.dot(normal, normals) < 0)
            return OrientedBoundary(facets, ori)
        return facets

    def facets_around(self, elements, flip=False) -> OrientedBoundary:
        """Return the oriented set of facets around a set of elements.

        Parameters
        ----------
        elements
            An array of element indices or, alternatively, something that can
            be cast into one via ``Mesh.normalize_elements``.
        flip
            If ``True``, use traces outside the subdomain and inward normals.

        """
        elements = self.normalize_elements(elements)
        facets, counts = np.unique(self.t2f[:, elements], return_counts=True)
        facets = facets[counts == 1]
        if flip:
            ori = (np.nonzero(~np.isin(self.f2t[:, facets], elements).T)[1]
                   .astype(np.int32))
        else:
            ori = (np.nonzero(np.isin(self.f2t[:, facets], elements).T)[1]
                   .astype(np.int32))
        return OrientedBoundary(facets, ori)

    def elements_satisfying(self,
                            test: Callable[[ndarray], ndarray]) -> ndarray:
        """Return elements whose midpoints satisfy some condition.

        Parameters
        ----------
        test
            A function which returns ``True`` for the element midpoints that
            are to be included in the return set.

        """
        midp = self.p[:, self.t].mean(axis=1)
        return np.nonzero(test(midp))[0].astype(np.int32)

    def _expand_facets(self, ix: ndarray) -> Tuple[ndarray, ndarray]:
        """Return vertices and edges corresponding to given facet indices.

        Parameters
        ----------
        ix
            An array of facet indices.

        """
        vertices = np.unique(self.facets[:, ix].flatten())

        if self.dim() == 3 and self.bndelem is not None:
            edges = np.unique(self.f2e[:, ix])
        else:
            edges = np.array([], dtype=np.int32)

        return vertices, edges

    def _mapping(self):
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

    def mapping(self):
        """Return a default reference mapping for the mesh."""
        return self._mapping()

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
        self.t = np.asarray(self.t, dtype=np.int32, order="K")

        M = self.elem.refdom.nnodes

        if self.nnodes > M and self.elem is not Element:
            # this is for high-order meshes, input in a different format
            # reorder DOFs to the expected format: vertex DOFs are first
            # note: not run if elem is not set
            p, t = self.doflocs, self.t
            t_nodes = t[:M]
            uniq, ix = np.unique(t_nodes, return_inverse=True)
            self.t = (np.arange(len(uniq), dtype=np.int32)[ix]
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
            if self.doflocs.shape[1] > 1e3:
                logger.warning("Transforming over 1000 vertices "
                               "to C_CONTIGUOUS.")
            self.doflocs = np.ascontiguousarray(self.doflocs)

        if not self.t.flags['C_CONTIGUOUS']:
            if self.t.shape[1] > 1e3:
                logger.warning("Transforming over 1000 elements "
                               "to C_CONTIGUOUS.")
            self.t = np.ascontiguousarray(self.t)

        # normalize data types
        # if self._boundaries is not None:
        #     self._boundaries = {
        #         k: v.astype(np.int32)
        #         for k, v in self._boundaries.items()
        #     }
        # if self._subdomains is not None:
        #     self._subdomains = {
        #         k: v.astype(np.int32)
        #         for k, v in self._subdomains.items()
        #     }

        # run validation
        if self.validate and logger.getEffectiveLevel() <= logging.DEBUG:
            self.is_valid()

    def is_valid(self, raise_=False) -> bool:
        """Perform expensive mesh validation.

        Parameters
        ----------
            raise_: raise an exception if the mesh is invalid.

        Returns
        -------
            bool: True if the mesh is valid.
        """
        logger.debug("Running mesh validation.")

        # check that the shape of doflocs and t are correct
        if self.doflocs.shape[0] != self.elem.refdom.dim():
            msg = "Mesh.doflocs, the point array, has incorrect shape."
            if raise_:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        if self.t.shape[0] != self.elem.refdom.nnodes:
            msg = "Mesh.t, the element connectivity, has incorrect shape."
            if raise_:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        # check that there are no duplicate points
        tmp = np.ascontiguousarray(self.p.T)
        p_unique = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]))
        if self.p.shape[1] != p_unique.shape[0]:
            msg = "Mesh contains duplicate vertices."
            if raise_:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        # check that all points are at least in some element
        if len(np.setdiff1d(np.arange(self.p.shape[1]),
                            np.unique(self.t))) > 0:
            msg = "Mesh contains a vertex not belonging to any element."
            if raise_:
                raise ValueError(msg)
            logger.debug(msg)
            return False

        logger.debug("Mesh validation completed with no warnings.")
        return True

    @staticmethod
    def _remove_duplicate_nodes(p, t):
        tmp = np.ascontiguousarray(p.T)
        tmp, ixa, ixb = np.unique(tmp.view([('', tmp.dtype)] * tmp.shape[1]),
                                  return_index=True, return_inverse=True)
        return p[:, ixa], Mesh._squeeze_if(ixb[t])

    @staticmethod
    def _squeeze_if(arr):
        # Workaround for the additional dimension introduced in
        # numpy 2.0 for the output of np.unique when using
        # return_index=True
        if len(arr.shape) > 2:
            return arr.squeeze(axis=2)
        return arr

    def __iter__(self):
        return iter((self.doflocs, self.t))

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
                cls(p, self._squeeze_if(ixb[self.t])),
                *[type(m)(p, self._squeeze_if(ixb[m.t + self.p.shape[1]]))
                  for i, m in enumerate(other)],
            ]
        raise NotImplementedError

    def __add__(self, other):
        """Join two meshes."""
        cls = type(self)
        if not isinstance(other, cls):
            raise TypeError("Can only join meshes with same type.")
        p = np.hstack((self.p.round(decimals=8),
                       other.p.round(decimals=8)))
        t = np.hstack((self.t, other.t + self.p.shape[1]))
        return cls(*self._remove_duplicate_nodes(p, t))

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
             filename: Union[str, PathLike],
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
    def from_dict(cls, data):

        if 'p' not in data or 't' not in data:
            raise ValueError("Dictionary must contain keys 'p' and 't'.")
        else:
            data['p'] = np.ascontiguousarray(np.array(data['p']).T)
            data['t'] = np.ascontiguousarray(np.array(data['t']).T)
        if 'boundaries' in data and data['boundaries'] is not None:
            data['boundaries'] = {k: np.array(v)
                                  for k, v in data['boundaries'].items()}
        if 'subdomains' in data and data['subdomains'] is not None:
            data['subdomains'] = {k: np.array(v)
                                  for k, v in data['subdomains'].items()}
        data['doflocs'] = data.pop('p')
        data['_subdomains'] = data.pop('subdomains')
        data['_boundaries'] = data.pop('boundaries')
        return cls(**data)

    def to_dict(self):

        boundaries = None
        subdomains = None
        if self.boundaries is not None:
            boundaries = {k: v.tolist() for k, v in self.boundaries.items()}
        if self.subdomains is not None:
            subdomains = {k: v.tolist() for k, v in self.subdomains.items()}
        return {
            'p': self.p.T.tolist(),
            't': self.t.T.tolist(),
            'boundaries': boundaries,
            'subdomains': subdomains,
        }

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
        return cls(cls.elem.refdom.p, cls.elem.refdom.t, validate=False)

    def copy(self):
        return replace(self)

    def morphed(self, *args):
        """Morph the mesh using functions.

        Parameters
        ----------
        funcs
            One function per dimension, input is `p` and output is the
            new coordinate of `p[i]', with `i=1,..,ndim`.

        """
        p = self.p.copy()
        for i, arg in enumerate(args):
            if arg is None:
                continue
            p[i] = arg(self.p)
        return replace(self, doflocs=p)

    def refined(self, times_or_ix: Union[int, ndarray] = 1):
        """Return a refined mesh.

        Parameters
        ----------
        times_or_ix
            Either an integer giving the number of uniform refinements or an
            array of element indices for adaptive refinement.

        """
        m = self
        has_boundaries = self.boundaries is not None
        has_subdomains = self.subdomains is not None
        if isinstance(times_or_ix, int):
            for _ in range(times_or_ix):
                mtmp = m._uniform()
                # fix subdomains for remaining mesh types
                if m._subdomains is not None and mtmp._subdomains is None:
                    N = int(mtmp.t.shape[1] / m.t.shape[1])
                    new_t = np.zeros((N, m.t.shape[1]), dtype=np.int32)
                    new_t[0] = np.arange(m.t.shape[1], dtype=np.int32)
                    for itr in range(N - 1):
                        new_t[itr + 1] = new_t[itr] + m.t.shape[1]
                    mtmp = replace(
                        mtmp,
                        _subdomains={
                            name: np.sort(new_t[:, ixs].flatten())
                            for name, ixs in m._subdomains.items()
                        },
                    )
                m = mtmp
        else:
            m = m._adaptive(times_or_ix)
        if has_boundaries and m.boundaries is None:
            logger.warning("Named boundaries invalidated by a call to "
                           "Mesh.refined()")
        if has_subdomains and self.subdomains is None:
            logger.warning("Named subdomains invalidated by a call to "
                           "Mesh.refined()")
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

    def smoothed(self, fixed_nodes=None):
        """Laplacian smoothing.

        Parameters
        ----------
        fixed_nodes
            A list of node indices that do not move.  By default, the boundary
            nodes are fixed.

        """

        if fixed_nodes is None:
            fixed_nodes = self.boundary_nodes()

        p = np.zeros(self.doflocs.shape)
        nv = p.shape[1]

        edges = self.edges if p.shape[0] == 3 else self.facets
        nneighbors = np.bincount(edges.reshape(-1), minlength=nv)

        p += np.array([np.bincount(edges[0], pi, minlength=nv)
                       for pi in self.doflocs[:, edges[1]]])
        p += np.array([np.bincount(edges[1], pi, minlength=nv)
                       for pi in self.doflocs[:, edges[0]]])
        p /= nneighbors

        p[:, fixed_nodes] = self.doflocs[:, fixed_nodes]

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

        # Always creates a mesh with duplicate verticies. Use validate=False to
        # suppress confusing logger DEBUG messages.
        return cls(p, t, validate=False)

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

        inverse = np.zeros((2, np.max(mapping) + 1), dtype=np.int32)
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

    def _reix(self, ix: ndarray) -> Tuple[ndarray, ndarray, ndarray]:
        """Connect ``self.p`` based on the indices ``ix``."""
        ixuniq = np.unique(ix)
        t = np.zeros(np.max(ix) + 1, dtype=np.int32)
        t[ixuniq] = np.arange(len(ixuniq), dtype=np.int32)
        return (
            np.ascontiguousarray(self.p[:, ixuniq]),
            np.ascontiguousarray(t[ix]),
            ixuniq
        )

    def trace(self, facets, mtype=None, project=None):
        """Create a trace mesh.

        Parameters
        ----------
        facets
            Criteria of which facets to include.  This input is normalized
            using ``self.normalize_facets``.
        mtype
            Optional subtype of Mesh which is used to initialize the return
            value.  If not provided, the raw Mesh type is used.
        project
            Optional lambda for modifying doflocs before initializing.  Usually
            for projecting doflocs because trace may lead to a lower
            dimensional mesh.  Useful example is ``lambda p: p[1:]``.

        """
        facets = self.normalize_facets(facets)
        p, t, _ = self._reix(self.facets[:, facets])
        return (Mesh if mtype is None else mtype)(
            (project(p) if project is not None else p),
            t
        ), facets

    def restrict(self,
                 elements,
                 return_mapping=False,
                 skip_boundaries=False,
                 skip_subdomains=False):
        """Restrict the mesh to a subset of elements.

        Parameters
        ----------
        elements
            Criteria of which elements to include.  This input is normalized
            using ``self.normalize_elements``.
        return_mapping
            Optionally, return the index mapping for vertices.
        skip_boundaries
            Optionally, skip retagging boundaries.
        skip_subdomains
            Optionally, skip retagging subdomains.

        """
        elements = self.normalize_elements(elements)
        p, t, ix = self._reix(self.t[:, elements])

        new_subdomains = None
        if not skip_subdomains and self.subdomains is not None:
            # map from old to new element index
            newt = np.zeros(self.t.shape[1], dtype=np.int32) - 1
            newt[elements] = np.arange(len(elements), dtype=np.int32)
            # remove 'elements' from each subdomain and remap
            new_subdomains = {
                k: newt[np.intersect1d(self.subdomains[k],
                                       elements).astype(np.int32)]
                for k in self.subdomains
            }

        new_boundaries = None
        if not skip_boundaries and self.boundaries is not None:
            # map from old to new facet index
            newf = np.zeros(self.facets.shape[1], dtype=np.int32) - 1
            facets = np.unique(self.t2f[:, elements])
            newf[facets] = np.arange(len(facets), dtype=np.int32)
            new_boundaries = {k: newf[self.boundaries[k]]
                              for k in self.boundaries}
            # filter facets not existing in the new mesh, value is -1
            new_boundaries = {k: v[v >= 0]
                              for k, v in new_boundaries.items()}

        out = replace(
            self,
            doflocs=p,
            t=t,
            _boundaries=new_boundaries,
            _subdomains=new_subdomains,
        )

        if return_mapping:
            return out, ix
        return out

    def remove_elements(self, elements):
        """Construct a new mesh by removing elements.

        Parameters
        ----------
        elements
            Criteria of which elements to include.  This input is normalized
            using ``self.normalize_elements``.

        """
        elements = self.normalize_elements(elements)
        return self.restrict(np.setdiff1d(np.arange(self.t.shape[1],
                                                    dtype=np.int32),
                                          elements))

    def remove_unused_nodes(self):
        p, t, _ = self._reix(self.t)
        return replace(
            self,
            doflocs=p,
            t=t,
        )

    def remove_duplicate_nodes(self):
        p, t = self._remove_duplicate_nodes(self.doflocs,
                                            self.t)
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

    def draw(self, visuals='matplotlib', **kwargs):
        """Convenience wrapper for skfem.visuals."""
        if not isinstance(visuals, str):
            logger.warning("First argument, 'visuals', must be a string.")
        mod = importlib.import_module('skfem.visuals.{}'.format(visuals))
        return mod.draw(self, **kwargs)

    def plot(self, x, visuals='matplotlib', **kwargs):
        """Convenience wrapper for skfem.visuals."""
        if not isinstance(visuals, str):
            logger.warning("Second argument, 'visuals', must be a string.")
        mod = importlib.import_module('skfem.visuals.{}'.format(visuals))
        return mod.plot(self, x, **kwargs)

    def normalize_nodes(self, nodes) -> ndarray:
        """Generate an array of node indices.

        Parameters
        ----------
        nodes
            Criteria of which nodes to include.  Function has different
            behavior based on the type of this parameter.

        """
        if isinstance(nodes, tuple):
            return self.normalize_nodes(
                lambda x: np.linalg.norm(x - np.array(list(nodes))[:, None],
                                         axis=0) < 1e-12
            )
        if isinstance(nodes, ndarray):
            # assumed an array of nodes
            return nodes
        elif isinstance(nodes, (list, set)):
            # Recurse over the list, building an array of all matching elements
            return np.unique(
                np.concatenate(
                    [self.normalize_nodes(n) for n in nodes]
                )
            )
        elif callable(nodes):
            return self.nodes_satisfying(nodes)
        raise NotImplementedError

    def normalize_facets(self, facets) -> ndarray:
        """Generate an array of facet indices.

        Parameters
        ----------
        facets
            Criteria of which facets to include.  Function has different
            behavior based on the type of this parameter.

        """
        if isinstance(facets, int):
            # Make  normalize_facets([1,2,3]) have the same behavior as
            # normalize_facets(np.array([1,2,3]))
            return np.array([facets])
        if isinstance(facets, ndarray):
            # Assume the facets have already been normalized
            return facets
        if facets is None:
            # Default behavior.
            return self.boundary_facets()
        elif isinstance(facets, (tuple, list, set)):
            # Recurse over the list, building an array of all matching facets
            return np.unique(
                np.concatenate(
                    [self.normalize_facets(f) for f in facets]
                )
            )
        elif callable(facets):
            # The callable should accept an array of facet centers and return
            # an boolean array with True for facets that should be included.
            return self.facets_satisfying(facets)
        elif isinstance(facets, str):
            # Assume string is the label of a boundary in the mesh.
            if ((self.boundaries is not None
                 and facets in self.boundaries)):
                return self.boundaries[facets]
            else:
                raise ValueError("Boundary '{}' not found.".format(facets))
        raise NotImplementedError

    def normalize_elements(self, elements) -> ndarray:
        """Generate an array of element indices.

        Parameters
        ----------
        elements
            Criteria of which elements to include.  Function has different
            behavior based on the type of this parameter.

        """
        if isinstance(elements, bool) and elements:
            return np.arange(self.nelements, dtype=np.int32)
        if isinstance(elements, int):
            # Make  normalize_elements([1,2,3]) have the same behavior as
            # normalize_elements(np.array([1,2,3]))
            return np.array([elements])
        if isinstance(elements, ndarray):
            # Assume the elements have already been normalized
            return elements
        if callable(elements):
            # The callable should accept an array of element centers and return
            # an boolean array with True for elements that should be included.
            return self.elements_satisfying(elements)
        elif isinstance(elements, (tuple, list, set)):
            # Recurse over the list, building an array of all matching elements
            return np.unique(
                np.concatenate(
                    [self.normalize_elements(e) for e in elements]
                )
            )
        elif isinstance(elements, str):
            # Assume string is the label of a subdomain in the mesh.
            if ((self.subdomains is not None
                 and elements in self.subdomains)):
                return self.subdomains[elements]
            else:
                raise ValueError("Subdomain '{}' not found.".format(elements))
        raise NotImplementedError
