import numpy as np

from skfem.mesh import MeshLine, MeshTri
from skfem.quadrature import get_quadrature


def intersect(m1, m2):
    """Create a supermesh between two nonmatching (1D or 2D) meshes.

    Parameters
    ----------
    m1
        The first mesh.
    m2
        The second mesh.

    Returns
    -------
    supermesh
        Mesh object with line (1D) or triangle (2D) elements.
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
    raise NotImplementedError("The given mesh types not supported.")


def _intersect2d(p1, t1, p2, t2):
    """Two-dimensional supermesh using shapely and bruteforce."""
    try:
        from shapely.geometry import Polygon
        from shapely.strtree import STRtree
        from shapely.ops import triangulate
    except Exception:
        raise Exception("2D supermeshing requires the package 'shapely>=2'.")
    t = np.empty((3, 0))
    p = np.empty((2, 0))
    polys = [Polygon(p1[:, t1[:, itr]].T) for itr in range(t1.shape[1])]
    ixmap = {id(polys[itr]): itr for itr in range(t1.shape[1])}
    s = STRtree(polys)
    ix1, ix2 = [], []
    for jtr in range(t2.shape[1]):
        poly1 = Polygon(p2[:, t2[:, jtr]].T)
        result = s.query(Polygon(p2[:, t2[:, jtr]].T))
        if len(result) == 0:
            continue
        for poly2 in s.geometries.take(result):
            tris = triangulate(poly1.intersection(poly2))
            for tri in tris:
                p = np.hstack((p, np.vstack(tri.exterior.xy)[:, :-1]))
                diff = np.max(t) + 1 if t.shape[1] > 0 else 0
                t = np.hstack((t, np.array([[0], [1], [2]]) + diff))
                ix1.append(ixmap[id(poly2)])
                ix2.append(jtr)
    return (
        MeshTri(p, t),
        np.array(ix1, dtype=np.int32),
        np.array(ix2, dtype=np.int32),
    )


def _intersect1d(p1, t1, p2, t2):
    """One-dimensional supermesh."""
    # find unique supermesh facets by combining nodes from both sides
    p = np.concatenate((p1.flatten().round(decimals=10),
                        p2.flatten().round(decimals=10)))
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
