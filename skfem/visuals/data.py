"""Backend-independent sampling of meshes and finite element fields.

All visualization backends consume :class:`VisData`, which is created by
:func:`sample`.  The reference element is subdivided ``nrefs`` times, the
resulting points are mapped to every element and the fields are evaluated
there.  The points are stored per element, i.e. discontinuously, so that
discontinuous fields (DG, P0, gradients, ...) and curved geometries are
represented without any special treatment.  Coincident points can be found
through :attr:`VisData.weld`.

"""

import json
import base64
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from numpy import ndarray

from ..mesh import (Mesh, MeshLine1, MeshTri1, MeshQuad1, MeshTet1,
                    MeshHex1, MeshWedge1)
from ..refdom import RefLine, RefTri, RefQuad, RefTet, RefHex, RefWedge


logger = logging.getLogger(__name__)


# reference domain -> (linear mesh type, cell type name)
_LINEAR: Dict[Any, Tuple[Any, str]] = {
    RefLine: (MeshLine1, 'line'),
    RefTri: (MeshTri1, 'tri'),
    RefQuad: (MeshQuad1, 'quad'),
    RefTet: (MeshTet1, 'tet'),
    RefHex: (MeshHex1, 'hex'),
    RefWedge: (MeshWedge1, 'wedge'),
}


def _cyclic(P: ndarray, idx) -> List[int]:
    """Order the vertices ``idx`` of a planar polygon cyclically."""
    idx = np.asarray(idx)
    Q = P[:, idx]
    D = Q - Q.mean(axis=1, keepdims=True)
    if P.shape[0] == 2:
        u, v = np.array([1., 0.]), np.array([0., 1.])
    else:
        n = np.cross(D[:, 0], D[:, 1])
        if np.linalg.norm(n) < 1e-12:
            n = np.cross(D[:, 0], D[:, 2])
        n = n / np.linalg.norm(n)
        u = D[:, 0] / np.linalg.norm(D[:, 0])
        v = np.cross(n, u)
    ang = np.arctan2(v @ D, u @ D)
    return [int(idx[i]) for i in np.argsort(ang)]


def _subdivision(refdom, nrefs: int) -> Mesh:
    meshtype = _LINEAR[refdom][0]
    R = meshtype.init_refdom()
    if nrefs > 0:
        try:
            R = R.refined(nrefs)
        except Exception:
            logger.warning("Refinement not supported for {}; using nrefs=0."
                           .format(meshtype.__name__))
    return R


def _subdivision_edges(refdom, R: Mesh) -> ndarray:
    """Edges of the subdivision lying on the edges of the reference cell."""
    dim = refdom.p.shape[0]
    if dim == 1:
        return np.zeros((2, 0), dtype=np.int64)
    if dim == 2:
        return R.facets[:, R.boundary_facets()]
    E = R.edges
    ref_e = np.array(refdom.edges).T
    A, B = refdom.p[:, ref_e[0]], refdom.p[:, ref_e[1]]
    keep = np.zeros(E.shape[1], dtype=bool)
    for k in range(A.shape[1]):
        a, d = A[:, k:k + 1], B[:, k:k + 1] - A[:, k:k + 1]
        ok = np.ones(E.shape[1], dtype=bool)
        for end in (0, 1):
            q = R.p[:, E[end]] - a
            t = (d[:, 0] @ q) / (d[:, 0] @ d[:, 0])
            r = q - d * t
            ok &= ((np.linalg.norm(r, axis=0) < 1e-9)
                   & (t > -1e-9) & (t < 1 + 1e-9))
        keep |= ok
    return E[:, keep]


def _degree(elem, refdom) -> int:
    """Rough polynomial degree of an element along an edge."""
    maxdeg = getattr(elem, 'maxdeg', 1)
    if refdom in (RefQuad, RefHex):
        return int(np.ceil(maxdeg / refdom.dim()))
    if refdom is RefWedge:
        return int(np.ceil(maxdeg / 2))
    return int(maxdeg)


def _default_nrefs(degree: int) -> int:
    if degree <= 1:
        return 0
    if degree == 2:
        return 1
    if degree <= 4:
        return 2
    return 3


def _weld(P: ndarray) -> ndarray:
    """Index of the unique point for each (approximately) coincident point."""
    if P.shape[0] == 0:
        return np.zeros(0, dtype=np.int64)
    span = float(np.ptp(P, axis=0).max()) or 1.0
    q = np.round((P - P.min(axis=0)) / span * (2 ** 20 - 1)).astype(np.int64)
    key = q[:, 0]
    for k in range(1, P.shape[1]):
        key = (key << 21) | q[:, k]
    _, W = np.unique(key, return_inverse=True)
    return W.reshape(-1)


@dataclass
class VisData:
    """Sampled mesh and fields, the common input of all backends.

    Attributes
    ----------
    points
        ``(npoints, gdim)`` point coordinates.  Each element owns ``nloc``
        consecutive points, i.e. the points are discontinuous.
    cells
        ``(ncells, nverts)`` linear sub-cells in the vertex order of the
        corresponding linear skfem mesh type.  Each element owns ``nsub``
        consecutive sub-cells.
    cell_type
        One of ``'line'``, ``'tri'``, ``'quad'``, ``'tet'``, ``'hex'``,
        ``'wedge'``.
    edges
        ``(nedges, 2)`` wireframe segments, i.e. the edges of the original
        elements (subdivided, so that they follow curved geometries).
    element
        ``(ncells,)`` index of the original element of each sub-cell.
    weld
        ``(npoints,)`` index of a unique point for each point; coincident
        points share the index.
    point_data
        Field name to ``(npoints,)`` or ``(npoints, ncomp)`` values.
    field_shapes
        Field name to the tensor shape of a single value, e.g. ``()`` for
        scalars, ``(2,)`` for vectors in 2D.
    meta
        Additional information (mesh and element names, nrefs, ...).

    """

    points: ndarray
    cells: ndarray
    cell_type: str
    edges: ndarray
    element: ndarray
    weld: ndarray
    nloc: int
    nsub: int
    refpoints: ndarray
    point_data: Dict[str, ndarray] = field(default_factory=dict)
    field_shapes: Dict[str, Tuple[int, ...]] = field(default_factory=dict)
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def dim(self) -> int:
        """Topological dimension of the cells."""
        return int(self.refpoints.shape[0])

    @property
    def gdim(self) -> int:
        """Dimension of the ambient space."""
        return int(self.points.shape[1])

    @property
    def npoints(self) -> int:
        return int(self.points.shape[0])

    @property
    def ncells(self) -> int:
        return int(self.cells.shape[0])

    @property
    def nelements(self) -> int:
        return int(self.points.shape[0] // self.nloc)

    @property
    def fields(self) -> List[str]:
        return list(self.point_data)

    @property
    def meshtype(self):
        return {v[1]: v[0] for v in _LINEAR.values()}[self.cell_type]

    def scalar(self, name: Optional[str] = None,
               component: Any = None) -> ndarray:
        """Return one scalar value per point.

        Parameters
        ----------
        name
            Field name; defaults to the first field.
        component
            ``None`` (scalar fields; magnitude for others), ``'magnitude'``,
            or an integer index to the flattened components.

        """
        if name is None:
            if not self.point_data:
                raise ValueError("VisData has no fields.")
            name = self.fields[0]
        v = self.point_data[name]
        if v.ndim == 1:
            return v
        if component is None or component in ('mag', 'magnitude'):
            return np.sqrt(np.sum(v ** 2, axis=1))
        return v[:, component]

    def triangles(self) -> ndarray:
        """Triangulate two-dimensional cells, ``(ntris, 3)``."""
        if self.dim != 2:
            raise ValueError("Only two-dimensional cells can be "
                             "triangulated.")
        poly = _cyclic(self.meshtype.init_refdom().p,
                       np.arange(self.cells.shape[1]))
        tris = [self.cells[:, [poly[0], poly[j], poly[j + 1]]]
                for j in range(1, len(poly) - 1)]
        return np.vstack(tris)

    def faces(self) -> Tuple[ndarray, ndarray]:
        """Triangulated faces of three-dimensional cells.

        Returns
        -------
        tris
            ``(nfaces * k, 3)`` triangles.
        face
            ``(nfaces * k,)`` face index of each triangle; the faces of the
            sub-cell ``c`` are ``c * nfaces_per_cell + j``.

        """
        R = self.meshtype.init_refdom()
        refdom = R.refdom
        tris, face = [], []
        nf = len(refdom.facets)
        for j, f in enumerate(refdom.facets):
            fl = _cyclic(R.p, np.array(f))
            for k in range(1, len(fl) - 1):
                tris.append(self.cells[:, [fl[0], fl[k], fl[k + 1]]])
                face.append(np.arange(self.ncells) * nf + j)
        return np.vstack(tris), np.concatenate(face)

    def boundary_triangles(self, cells: Optional[ndarray] = None) -> ndarray:
        """Triangles on the boundary of a set of three-dimensional cells.

        Parameters
        ----------
        cells
            Optional boolean mask or index array of sub-cells to include.

        """
        R = self.meshtype.init_refdom()
        mask = np.ones(self.ncells, dtype=bool)
        if cells is not None:
            mask = np.zeros(self.ncells, dtype=bool)
            mask[cells] = True
        C = self.cells[mask]
        out = []
        keys = [np.sort(self.weld[C[:, f]], axis=1)
                for f in R.refdom.facets]
        nverts = [len(f) for f in R.refdom.facets]
        # find faces appearing only once, separately per face size
        for n in sorted(set(nverts)):
            js = [j for j in range(len(nverts)) if nverts[j] == n]
            K = np.vstack([keys[j] for j in js])
            _, inv, cnt = np.unique(K, axis=0, return_inverse=True,
                                    return_counts=True)
            once = cnt[inv.reshape(-1)] == 1
            offset = 0
            for j in js:
                sel = once[offset:offset + C.shape[0]]
                offset += C.shape[0]
                fl = _cyclic(R.p, np.array(R.refdom.facets[j]))
                for k in range(1, len(fl) - 1):
                    out.append(C[sel][:, [fl[0], fl[k], fl[k + 1]]])
        if not out:
            return np.zeros((0, 3), dtype=np.int64)
        return np.vstack(out)

    def to_mesh(self) -> Mesh:
        """Return a (discontinuous) linear skfem mesh of the sub-cells."""
        return self.meshtype(np.ascontiguousarray(self.points.T),
                             np.ascontiguousarray(self.cells.T))

    def to_meshio(self, **kwargs):
        """Convert to :class:`meshio.Mesh` including the fields."""
        from skfem.io.meshio import to_meshio
        point_data = {k: v for k, v in self.point_data.items()}
        point_data.update(kwargs.pop('point_data', {}))
        return to_meshio(self.to_mesh(), point_data=point_data, **kwargs)

    def save(self, filename, **kwargs) -> None:
        """Write the sampled mesh and fields using meshio."""
        self.to_meshio(**kwargs).write(filename)

    def to_dict(self, dtype=np.float32) -> Dict[str, Any]:
        """Compact JSON-serializable representation.

        The arrays are base64-encoded little-endian buffers (``float32`` for
        coordinates and values, ``int32`` for indices).  This is the input
        format of the web viewer.

        """
        def b64(a, t):
            return base64.b64encode(
                np.ascontiguousarray(a, dtype=np.dtype(t).newbyteorder('<'))
                .tobytes()).decode('ascii')

        fields = []
        for name, v in self.point_data.items():
            shape = tuple(self.field_shapes.get(name, v.shape[1:]))
            v2 = v.reshape(v.shape[0], -1)
            ncomp = v2.shape[1]
            ranges = []
            for k in range(ncomp):
                ranges.append(_range(v2[:, k]))
            item = {
                'name': name,
                'shape': list(shape),
                'ncomp': ncomp,
                'data': b64(v2, dtype),
                'range': ranges,
            }
            if ncomp > 1:
                item['magrange'] = _range(np.sqrt(np.sum(v2 ** 2, axis=1)))
            fields.append(item)

        meta = dict(self.meta)
        if self.dim == 2:
            meta['poly'] = _cyclic(self.meshtype.init_refdom().p,
                                   np.arange(self.cells.shape[1]))
        elif self.dim == 3:
            R = self.meshtype.init_refdom()
            meta['faces'] = [_cyclic(R.p, np.array(f))
                             for f in R.refdom.facets]

        return {
            'version': 1,
            'cell_type': self.cell_type,
            'dim': self.dim,
            'gdim': self.gdim,
            'nloc': self.nloc,
            'nsub': self.nsub,
            'nel': self.nelements,
            'nvc': int(self.cells.shape[1]),
            'points': b64(self.points, dtype),
            'cells': b64(self.cells, np.int32),
            'edges': b64(self.edges, np.int32),
            'weld': b64(self.weld, np.int32),
            'fields': fields,
            'meta': meta,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict())


def _range(v: ndarray) -> List[float]:
    fin = v[np.isfinite(v)]
    if fin.size == 0:
        return [0., 1.]
    return [float(fin.min()), float(fin.max())]


def _is_basis(obj) -> bool:
    from ..assembly.basis import AbstractBasis
    return isinstance(obj, AbstractBasis)


def _normalize_fields(obj, x) -> List[Tuple[str, Any, Any, Any]]:
    """Return a list of (name, basis, array, func)."""
    from ..assembly.basis.composite_basis import CompositeBasis
    from ..element import ElementComposite

    default_basis = obj if _is_basis(obj) else None

    if x is None:
        return []
    if not isinstance(x, dict):
        x = {'u': x}

    out: List[Tuple[str, Any, Any, Any]] = []
    for name, spec in x.items():
        basis: Any = default_basis
        arr: Any = None
        func: Any = None
        if isinstance(spec, tuple):
            for item in spec:
                if _is_basis(item):
                    basis = item
                elif callable(item):
                    func = item
                else:
                    arr = np.asarray(item)
        elif callable(spec):
            func = spec
        else:
            arr = np.asarray(spec)

        if basis is None and arr is not None:
            # plain mesh: nodal or elementwise values
            mesh = obj
            if arr.shape[-1] == mesh.nvertices:
                basis = 'vertex'
            elif arr.shape[-1] == mesh.doflocs.shape[1]:
                basis = 'nodal'
            elif arr.shape[-1] == mesh.nelements:
                basis = 'element'
            else:
                raise ValueError("Array '{}' of length {} matches neither "
                                 "the number of vertices, nodes nor "
                                 "elements of the mesh."
                                 .format(name, arr.shape[-1]))
            out.append((name, basis, arr, func))
            continue

        if basis is not None and arr is None and func is not None:
            raise ValueError("A callable needs an array to act on; use "
                             "a tuple (array, callable).")

        if isinstance(basis, CompositeBasis) or (
                _is_basis(basis)
                and isinstance(getattr(basis, 'elem', None),
                               ElementComposite)):
            if func is not None:
                raise ValueError("Callables are not supported for "
                                 "composite bases; split the solution.")
            for i, (a, b) in enumerate(basis.split(arr)):
                out.append(("{}{}".format(name, i), b, a, None))
            continue

        if arr is not None and _is_basis(basis) and arr.shape[0] != basis.N:
            raise ValueError("Array '{}' has length {} but basis has {} "
                             "DOFs.".format(name, arr.shape[0], basis.N))
        out.append((name, basis, arr, func))
    return out


def sample(obj,
           x: Any = None,
           nrefs: Optional[int] = None,
           elements: Any = None,
           **kwargs) -> VisData:
    """Sample a mesh and finite element fields for visualization.

    Parameters
    ----------
    obj
        A :class:`~skfem.mesh.Mesh` or a :class:`~skfem.assembly.CellBasis`
        (or :class:`~skfem.assembly.basis.CompositeBasis`).
    x
        The field(s).  One of

        - an array of DOFs of the basis ``obj``,
        - for a mesh, an array of nodal values (length ``nvertices``) or
          of elementwise constants (length ``nelements``),
        - a dict from field names to any of the above or to tuples
          ``(basis, array)``, ``(array, func)`` or ``(basis, array, func)``
          where ``func`` acts on the interpolated
          :class:`~skfem.element.DiscreteField`, e.g.
          ``lambda w: w.grad[0]``.

    nrefs
        Number of subdivisions of the reference element.  By default chosen
        based on the polynomial degree of the elements and the mesh.
    elements
        Optional subset of elements to sample, as accepted by
        :meth:`~skfem.mesh.Mesh.normalize_elements`.

    """
    from ..assembly import CellBasis
    from ..assembly.basis.composite_basis import CompositeBasis

    if isinstance(obj, CompositeBasis):
        mesh = obj.bases[0].mesh
        mapping = obj.bases[0].mapping
        tind_basis = obj.bases[0].tind
    elif _is_basis(obj):
        mesh = obj.mesh
        mapping = obj.mapping
        tind_basis = getattr(obj, 'tind', None)
    elif isinstance(obj, Mesh):
        mesh = obj
        mapping = mesh._mapping()
        tind_basis = None
    else:
        raise TypeError("Cannot sample object of type {}.".format(type(obj)))

    refdom = mesh.refdom
    if refdom not in _LINEAR:
        raise NotImplementedError("Reference domain {} not supported."
                                  .format(refdom.__name__))
    meshtype, cell_type = _LINEAR[refdom]

    if elements is not None:
        tind = mesh.normalize_elements(elements)
    elif tind_basis is not None:
        tind = np.asarray(tind_basis)
    else:
        tind = np.arange(mesh.nelements)

    fields = _normalize_fields(obj, x)

    if nrefs is None:
        deg = _degree(mesh.elem, refdom)
        for (_, b, _, _) in fields:
            if _is_basis(b):
                deg = max(deg, _degree(b.elem, refdom))
        nrefs = _default_nrefs(deg)

    R = _subdivision(refdom, int(nrefs))
    X = R.p
    if kwargs.get('_shrink', False):
        # used by refinterp in 1D for backwards compatibility
        X = (1. - 1e-10) * X + 5e-11
    nloc, nsub, nel = X.shape[1], R.t.shape[1], len(tind)

    pts = mapping.F(X, tind=tind)  # (gdim, nel, nloc)
    gdim = pts.shape[0]
    P = np.ascontiguousarray(pts.transpose(1, 2, 0).reshape(-1, gdim))
    C = (R.t.T[None, :, :]
         + nloc * np.arange(nel)[:, None, None]).reshape(-1, R.t.shape[0])
    Es = _subdivision_edges(refdom, R)
    E = (Es.T[None, :, :]
         + nloc * np.arange(nel)[:, None, None]).reshape(-1, 2)

    point_data = {}
    field_shapes = {}
    elem_names = {}
    quad = (X, np.ones(nloc))
    cache = {}
    for (name, b, arr, func) in fields:
        val: Any
        if isinstance(b, str) and b == 'vertex':
            # nodal values on a mesh: interpolate using the linear element
            lb = CellBasis(mesh, meshtype.elem(), mapping=mapping,
                           quadrature=quad, elements=tind)
            if arr.shape[0] != lb.N:
                # e.g. quadratic meshes: values on vertices only
                y = np.zeros(lb.N, dtype=arr.dtype)
                y[lb.nodal_dofs[0]] = arr
                arr = y
            val = np.asarray(lb.interpolate(arr))
            elem_names[name] = type(lb.elem).__name__
        elif isinstance(b, str) and b == 'nodal':
            # values at mesh.doflocs, e.g. quadratic meshes
            lb = CellBasis(mesh, mesh.elem(), mapping=mapping,
                           quadrature=quad, elements=tind)
            val = np.asarray(lb.interpolate(arr))
            elem_names[name] = type(lb.elem).__name__
        elif isinstance(b, str) and b == 'element':
            val = np.repeat(arr[tind][:, None], nloc, axis=1)
            elem_names[name] = 'elementwise constant'
        else:
            key = id(b)
            if key not in cache:
                cache[key] = CellBasis(mesh, b.elem, mapping=b.mapping,
                                       quadrature=quad, elements=tind)
            vb = cache[key]
            w = vb.interpolate(arr)
            val = func(w) if func is not None else w
            val = np.asarray(val)
            elem_names[name] = type(b.elem).__name__
        if val.shape[-2:] != (nel, nloc):
            raise ValueError("Field '{}' has unexpected shape {}."
                             .format(name, val.shape))
        shape = val.shape[:-2]
        val = val.reshape(-1, nel * nloc).T
        if np.iscomplexobj(val):
            logger.warning("Field '{}' is complex; only the real part is "
                           "visualized.".format(name))
            val = val.real
        if len(shape) == 0:
            val = val[:, 0]
        point_data[name] = np.ascontiguousarray(val, dtype=np.float64)
        field_shapes[name] = tuple(int(s) for s in shape)

    meta = {
        'mesh': type(mesh).__name__,
        'nvertices': int(mesh.nvertices),
        'nelements': int(mesh.nelements),
        'nrefs': int(nrefs),
        'elements': elem_names,
    }

    return VisData(
        points=P,
        cells=C.astype(np.int64),
        cell_type=cell_type,
        edges=E.astype(np.int64),
        element=np.repeat(tind, nsub),
        weld=_weld(P),
        nloc=nloc,
        nsub=nsub,
        refpoints=R.p,
        point_data=point_data,
        field_shapes=field_shapes,
        meta=meta,
    )
