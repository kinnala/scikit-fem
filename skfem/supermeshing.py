import numpy as np

from skfem.mesh import MeshLine, MeshTri, MeshTet
from skfem.quadrature import get_quadrature


def intersect(m1, m2):
    """Create a supermesh between two nonmatching meshes.

    Supports 1D meshes, 2D meshes with convex elements (triangles or
    quadrilaterals) and 3D tetrahedral meshes.

    Parameters
    ----------
    m1
        The first mesh.
    m2
        The second mesh.

    Returns
    -------
    supermesh
        Mesh object with line (1D), triangle (2D) or tetrahedral (3D)
        elements.
    ix1
        Index of each supermesh element within the first mesh.
    ix2
        Index of each supermesh element within the second mesh.

    """
    p1, t1 = m1
    p2, t2 = m2
    if p1.shape[0] == 1 and p2.shape[0] == 1:
        return _intersect1d(p1, t1, p2, t2)
    elif p1.shape[0] == 2 and p2.shape[0] == 2:
        return _intersect2d(p1, t1, p2, t2)
    elif (p1.shape[0] == 3 and p2.shape[0] == 3
          and t1.shape[0] == 4 and t2.shape[0] == 4):
        return _intersect3d(p1, t1, p2, t2)
    raise NotImplementedError("The given mesh types not supported.")


def _join(k1, i1, k2, i2):
    """Return all pairs (i1[a], i2[b]) such that k1[a] == k2[b]."""
    order = np.argsort(k2, kind='stable')
    k2, i2 = k2[order], i2[order]
    start = np.searchsorted(k2, k1, side='left')
    count = np.searchsorted(k2, k1, side='right') - start
    offset = np.cumsum(count) - count
    ix = (np.arange(count.sum())
          + np.repeat(start - offset, count))
    return np.repeat(i1, count), i2[ix]


def _bbox_pairs(lo1, hi1, lo2, hi2):
    """Find the pairs of overlapping axis-aligned bounding boxes.

    Vectorized loose quadtree (2D) or octree (3D): each box is placed on the
    level where the cell size is larger than the box, so that it overlaps at
    most 2**dim cells.  Two boxes can overlap only if they share a cell on the
    coarser of their two levels.

    Parameters
    ----------
    lo1, hi1
        Arrays of shape (dim, n1); the lower and upper corners of the boxes.
    lo2, hi2
        Arrays of shape (dim, n2).

    Returns
    -------
    i1, i2
        Indices of the overlapping pairs, sorted by i2 and then by i1.

    """
    dim = lo1.shape[0]
    origin = np.minimum(lo1.min(axis=1), lo2.min(axis=1))[:, None]
    size = np.max(np.maximum(hi1.max(axis=1), hi2.max(axis=1))
                  - origin[:, 0])
    size = size * (1 + 1e-10) if size > 0 else 1.
    maxlevel = 60 // dim

    def cells(lo, hi):
        extent = np.max(hi - lo, axis=0)
        with np.errstate(divide='ignore'):
            level = np.floor(np.log2(size / extent))
        level = np.clip(level, 0, maxlevel).astype(np.int64)
        n = 2 ** level
        h = size / n
        ilo = np.clip(np.floor((lo - origin) / h), 0, n - 1).astype(np.int64)
        ihi = np.clip(np.floor((hi - origin) / h), 0, n - 1).astype(np.int64)
        elem, lev, ijk = [], [], []
        for offset in np.ndindex(*(dim * (2,))):
            ix = ilo + np.array(offset)[:, None]
            keep = np.all(ix <= ihi, axis=0)
            elem.append(np.nonzero(keep)[0])
            lev.append(level[keep])
            ijk.append(ix[:, keep])
        return (np.concatenate(elem),
                np.concatenate(lev),
                np.concatenate(ijk, axis=1))

    def key(ijk, level, target):
        ijk = ijk >> (level - target)
        return sum(ijk[k] << (k * target) for k in range(dim))

    tol = 1e-12 * size

    def overlapping(i1, i2):
        keep = np.all((lo1[:, i1] < hi2[:, i2] - tol)
                      & (lo2[:, i2] < hi1[:, i1] - tol), axis=0)
        return i2[keep] * lo1.shape[1] + i1[keep]

    e1, l1, c1 = cells(lo1, hi1)
    e2, l2, c2 = cells(lo2, hi2)
    pairs = []
    for level in np.union1d(l1, l2):
        a, b = l1 == level, l2 >= level
        pairs.append(overlapping(*_join(key(c1[:, a], l1[a], level), e1[a],
                                        key(c2[:, b], l2[b], level), e2[b])))
        a, b = l1 > level, l2 == level
        pairs.append(overlapping(*_join(key(c2[:, b], l2[b], level), e2[b],
                                        key(c1[:, a], l1[a], level),
                                        e1[a])[::-1]))
    pairs = np.unique(np.concatenate(pairs))
    return pairs % lo1.shape[1], pairs // lo1.shape[1]


def _candidate_pairs(p1, t1, p2, t2):
    x1, x2 = p1[:, t1], p2[:, t2]
    return _bbox_pairs(x1.min(axis=1), x1.max(axis=1),
                       x2.min(axis=1), x2.max(axis=1))


def _chunks(n, size):
    for start in range(0, n, size):
        yield slice(start, min(start + size, n))


def _compact(X, ok):
    """Move the valid points (ok) to the front and drop the empty columns."""
    nok = ok.sum(axis=-1)
    order = np.argsort(~ok, axis=-1, kind='stable')
    order = order[..., :max(nok.max(initial=0), 3)]
    return np.take_along_axis(X, order[..., None], axis=-2), nok


def _cross2d(a, b):
    return a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]


def _convex_polygon_intersection(P, Q):
    """Intersect pairs of convex polygons and triangulate the result.

    Parameters
    ----------
    P
        Array of shape (n, nv1, 2); the vertices of n convex polygons in
        cyclic order (clockwise or counterclockwise).
    Q
        Array of shape (n, nv2, 2).

    Returns
    -------
    tris
        Array of shape (ntri, 3, 2); the triangles of the intersections.
    ix
        Array of shape (ntri,); index of the polygon pair of each triangle.

    """
    rP = np.roll(P, -1, axis=1) - P
    rQ = np.roll(Q, -1, axis=1) - Q
    areaP = .5 * _cross2d(P, rP).sum(axis=1)
    areaQ = .5 * _cross2d(Q, rQ).sum(axis=1)
    tol = 1e-12 * np.maximum(np.abs(areaP), np.abs(areaQ))

    def dist(X, R, rR, area):
        # scaled distances (n, m, nv) of X (n, m, 2) inside the edges of R
        c = _cross2d(rR[:, None], X[:, :, None] - R[:, None])
        return np.sign(area)[:, None, None] * c

    def inside(X, R, rR, area):
        return np.all(dist(X, R, rR, area) >= -tol[:, None, None], axis=2)

    # discard pairs separated by an edge of either polygon
    pair = np.nonzero(
        ~np.any(np.all(dist(Q, P, rP, areaP) <= tol[:, None, None], axis=1),
                axis=1)
        & ~np.any(np.all(dist(P, Q, rQ, areaQ) <= tol[:, None, None], axis=1),
                  axis=1))[0]
    P, Q, rP, rQ = P[pair], Q[pair], rP[pair], rQ[pair]
    areaP, areaQ, tol = areaP[pair], areaQ[pair], tol[pair]
    n = P.shape[0]

    # edge-edge intersections: P[i] + s rP[i] = Q[j] + t rQ[j]
    d = Q[:, None] - P[:, :, None]
    denom = _cross2d(rP[:, :, None], rQ[:, None])
    with np.errstate(divide='ignore', invalid='ignore'):
        s = _cross2d(d, rQ[:, None]) / denom
        t = _cross2d(d, rP[:, :, None]) / denom
        E = P[:, :, None] + s[..., None] * rP[:, :, None]
    eps = 1e-12
    Eok = ((np.abs(denom) > tol[:, None, None])
           & (s >= -eps) & (s <= 1 + eps)
           & (t >= -eps) & (t <= 1 + eps))

    X = np.concatenate((P, Q, E.reshape(n, -1, 2)), axis=1)
    ok = np.concatenate((inside(P, Q, rQ, areaQ),
                         inside(Q, P, rP, areaP),
                         Eok.reshape(n, -1)), axis=1)
    X, nok = _compact(X, ok)
    ok = np.arange(X.shape[1])[None] < nok[:, None]
    X[~ok] = 0.

    # the intersection is convex: sort the vertices by angle around their
    # centroid and fan triangulate
    c = X.sum(axis=1) / np.maximum(nok, 1)[:, None]
    angle = np.arctan2(X[..., 1] - c[:, 1:2], X[..., 0] - c[:, 0:1])
    angle[~ok] = np.inf
    X = np.take_along_axis(X, np.argsort(angle, axis=1)[..., None], axis=1)
    m = X.shape[1]
    k = np.arange(1, m - 1)
    tris = np.stack((np.broadcast_to(X[:, :1], (n, m - 2, 2)),
                     X[:, 1:-1],
                     X[:, 2:]), axis=2)
    area = .5 * np.abs(_cross2d(tris[:, :, 1] - tris[:, :, 0],
                                tris[:, :, 2] - tris[:, :, 0]))
    keep = (k[None] < nok[:, None] - 1) & (area > tol[:, None])
    return tris[keep], pair[np.nonzero(keep)[0]]


def _intersect2d(p1, t1, p2, t2, chunksize=10000):
    """Two-dimensional supermesh of convex (triangular or quadrilateral)
    elements using a vectorized quadtree search and convex polygon
    intersection."""
    i1, i2 = _candidate_pairs(p1, t1, p2, t2)
    tris, ix = [], []
    for s in _chunks(len(i1), chunksize):
        tr, k = _convex_polygon_intersection(p1[:, t1[:, i1[s]]].T,
                                             p2[:, t2[:, i2[s]]].T)
        tris.append(tr)
        ix.append(k + s.start)
    tris = np.concatenate(tris) if tris else np.empty((0, 3, 2))
    ix = np.concatenate(ix) if ix else np.empty(0, dtype=np.int64)
    return (
        MeshTri(np.ascontiguousarray(tris.reshape(-1, 2).T),
                np.arange(3 * len(tris)).reshape(3, -1, order='F')),
        i1[ix].astype(np.int32),
        i2[ix].astype(np.int32),
    )


def _tet_planes(X):
    """Outward unit normals (n, 4, 3) and offsets (n, 4) of tetrahedra X
    (n, 4, 3); face k is opposite to vertex k."""
    faces = np.array([[1, 2, 3], [0, 3, 2], [0, 1, 3], [0, 2, 1]])
    F = X[:, faces]
    N = np.cross(F[:, :, 1] - F[:, :, 0], F[:, :, 2] - F[:, :, 0])
    N /= np.linalg.norm(N, axis=2, keepdims=True)
    c = np.einsum('nki,nki->nk', N, F[:, :, 0])
    # flip to outward if the tetrahedron is negatively oriented
    sgn = np.sign(np.einsum('nki,nki->nk', N, X)[:, :1] - c[:, :1])
    return -sgn[..., None] * N, -sgn * c


def _tetrahedron_intersection(A, B):
    """Intersect pairs of tetrahedra and tetrahedralize the result.

    Parameters
    ----------
    A, B
        Arrays of shape (n, 4, 3).

    Returns
    -------
    tets
        Array of shape (ntet, 4, 3).
    ix
        Array of shape (ntet,); index of the pair of each tetrahedron.

    """
    edges = np.array([[0, 1], [0, 2], [0, 3], [1, 2], [1, 3], [2, 3]])
    NA, cA = _tet_planes(A)
    NB, cB = _tet_planes(B)
    h = np.maximum(np.linalg.norm(A[:, :1] - A[:, 1:], axis=2).max(axis=1),
                   np.linalg.norm(B[:, :1] - B[:, 1:], axis=2).max(axis=1))
    tol = 1e-12 * h

    def dist(X, N, c):
        # signed distances (n, m, 4) of points X (n, m, 3) to planes
        return np.einsum('nmi,nki->nmk', X, N) - c[:, None]

    # discard pairs separated by a face plane of either tetrahedron
    pair = np.nonzero(
        ~np.any(np.all(dist(B, NA, cA) >= -tol[:, None, None], axis=1),
                axis=1)
        & ~np.any(np.all(dist(A, NB, cB) >= -tol[:, None, None], axis=1),
                  axis=1))[0]
    A, B, h, tol = A[pair], B[pair], h[pair], tol[pair]
    NA, cA, NB, cB = NA[pair], cA[pair], NB[pair], cB[pair]
    n = A.shape[0]

    def edge_plane(X, N, c):
        # intersections of the edges of X with the planes (N, c)
        P0, P1 = X[:, edges[:, 0]], X[:, edges[:, 1]]
        d0, d1 = dist(P0, N, c), dist(P1, N, c)
        with np.errstate(divide='ignore', invalid='ignore'):
            s = d0 / (d0 - d1)
            E = P0[:, :, None] + s[..., None] * (P1 - P0)[:, :, None]
        ok = (np.abs(d0 - d1) > tol[:, None, None]) & (s >= 0) & (s <= 1)
        return E.reshape(n, -1, 3), ok.reshape(n, -1)

    EA, okA = edge_plane(A, NB, cB)
    EB, okB = edge_plane(B, NA, cA)
    X = np.concatenate((A, B, EA, EB), axis=1)
    dA, dB = dist(X, NA, cA), dist(X, NB, cB)
    ok = (np.concatenate((np.ones((n, 8), dtype=bool), okA, okB), axis=1)
          & np.all(dA <= tol[:, None, None], axis=2)
          & np.all(dB <= tol[:, None, None], axis=2))
    X, nok = _compact(X, ok)
    ok = np.arange(X.shape[1])[None] < nok[:, None]
    X[~ok] = 0.
    dA, dB = dist(X, NA, cA), dist(X, NB, cB)
    C = X.sum(axis=1) / np.maximum(nok, 1)[:, None]

    # the faces of the intersection lie on the planes of A and B; drop the
    # planes of B coinciding with a plane of A to avoid duplicate faces
    N = np.concatenate((NA, NB), axis=1)
    same = (np.all(np.abs(NB[:, :, None] - NA[:, None]) < 1e-10, axis=3)
            & (np.abs(cB[:, :, None] - cA[:, None]) < tol[:, None, None]))
    plane_ok = np.concatenate((np.ones((n, 4), dtype=bool),
                               ~np.any(same, axis=2)), axis=1)
    on = (ok[:, None]
          & (np.abs(np.concatenate((dA, dB), axis=2)) <= tol[:, None, None])
          .transpose(0, 2, 1)
          & plane_ok[..., None])  # (n, 8, m)
    Y, non = _compact(np.broadcast_to(X[:, None], on.shape + (3,)), on)
    on = np.arange(Y.shape[2])[None, None] < non[..., None]

    # sort the vertices of each face by angle around the face centroid
    U = np.cross(N, np.where(np.abs(N[..., :1]) < .9,
                             np.array([1., 0., 0.]),
                             np.array([0., 1., 0.])))
    U /= np.linalg.norm(U, axis=2, keepdims=True)
    V = np.cross(N, U)
    Fc = (np.where(on[..., None], Y, 0.).sum(axis=2)
          / np.maximum(non, 1)[..., None])
    R = Y - Fc[:, :, None]
    angle = np.arctan2(np.einsum('nfmi,nfi->nfm', R, V),
                       np.einsum('nfmi,nfi->nfm', R, U))
    angle[~on] = np.inf
    Y = np.take_along_axis(Y, np.argsort(angle, axis=2)[..., None], axis=2)

    # fan triangulate each face and connect to the centroid
    m = Y.shape[2]
    k = np.arange(1, m - 1)
    tets = np.stack((np.broadcast_to(C[:, None, None], (n, 8, m - 2, 3)),
                     np.broadcast_to(Y[:, :, :1], (n, 8, m - 2, 3)),
                     Y[:, :, 1:-1],
                     Y[:, :, 2:]), axis=3)
    vol = np.abs(np.einsum('nfki,nfki->nfk',
                           tets[:, :, :, 1] - tets[:, :, :, 0],
                           np.cross(tets[:, :, :, 2] - tets[:, :, :, 0],
                                    tets[:, :, :, 3] - tets[:, :, :, 0])))
    keep = ((k[None, None] < non[..., None] - 1)
            & (vol > (1e-10 * h ** 3)[:, None, None]))
    return tets[keep], pair[np.nonzero(keep)[0]]


def _intersect3d(p1, t1, p2, t2, chunksize=2000):
    """Three-dimensional supermesh of tetrahedral meshes using a vectorized
    octree search and convex polyhedron
    intersection."""
    i1, i2 = _candidate_pairs(p1, t1, p2, t2)
    tets, ix = [], []
    for s in _chunks(len(i1), chunksize):
        te, k = _tetrahedron_intersection(p1[:, t1[:, i1[s]]].T,
                                          p2[:, t2[:, i2[s]]].T)
        tets.append(te)
        ix.append(k + s.start)
    tets = np.concatenate(tets) if tets else np.empty((0, 4, 3))
    ix = np.concatenate(ix) if ix else np.empty(0, dtype=np.int64)
    return (
        MeshTet(np.ascontiguousarray(tets.reshape(-1, 3).T),
                np.arange(4 * len(tets)).reshape(4, -1, order='F')),
        i1[ix].astype(np.int32),
        i2[ix].astype(np.int32),
    )


def _intersect1d(p1, t1, p2, t2):
    """One-dimensional supermesh."""
    # Find unique supermesh facets by combining nodes from both
    # sides in the intersection of x-ranges.
    p1f = p1.flatten().round(decimals=10)
    p2f = p2.flatten().round(decimals=10)

    xmin = max(p1f.min(), p2f.min())
    xmax = min(p1f.max(), p2f.max())

    p1f = p1f[(p1f >= xmin) & (p1f <= xmax)]
    p2f = p2f[(p2f >= xmin) & (p2f <= xmax)]
    p = np.concatenate((p1f, p2f))
    p = np.unique(p)
    t = np.array([np.arange(len(p) - 1), np.arange(1, len(p))])
    p = np.array([p])

    supermap = MeshLine(p, t)._mapping()
    mps = supermap.F(np.array([[.5]]))
    ix1 = MeshLine(p1, t1).element_finder()(mps[0, :, 0])
    ix2 = MeshLine(p2, t2).element_finder()(mps[0, :, 0])

    return MeshLine(p, t), ix1, ix2


def elementwise_quadrature(mesh, supermesh=None, tind=None, intorder=None):
    """For creating element-by-element quadrature rules.

    Parameters
    ----------
    mesh
        The mesh for which to create the quadrature rules.
    supermesh
        Created using skfem.supermeshing.intersect.
    tind
        A subset of elements
    intorder
        Integration order, by default equal to 4.

    """
    if intorder is None:
        intorder = 4
    if supermesh is None:
        raise Exception("elementwise_quadrature: User must provide "
                        "'supermesh' keyword argument which has been "
                        "created using skfem.supermeshing.intersect.")
    X, W = get_quadrature(supermesh.elem, intorder)
    mmap = mesh.mapping()
    smap = supermesh.mapping()
    return (
        mmap.invF(smap.F(X), tind=tind),
        np.abs(smap.detDF(X) / mmap.detDF(X, tind=tind)) * W,
    )
