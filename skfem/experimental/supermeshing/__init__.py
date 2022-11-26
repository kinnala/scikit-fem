import numpy as np

from skfem.mesh import MeshLine, MeshTri
from skfem.quadrature import get_quadrature


def intersect2d(m1, m2):
    """Two-dimensional supermesh using shapely and bruteforce."""
    from shapely.geometry import Polygon
    from shapely.ops import triangulate
    p1, t1 = m1
    p2, t2 = m2
    t = np.empty((3, 0))
    p = np.empty((2, 0))
    ix1, ix2 = [], []
    for itr in range(m1.t.shape[1]):
        for jtr in range(m2.t.shape[1]):
            tri1, tri2 = p1[:, t1[:, itr]], p2[:, t2[:, jtr]]
            poly1 = Polygon(tri1.T)
            poly2 = Polygon(tri2.T)
            if not poly1.intersects(poly2):
                continue
            tris = triangulate(poly1.intersection(poly2))
            for tri in tris:
                p = np.hstack((p, np.vstack(tri.exterior.xy)[:, :-1]))
                diff = np.max(t) + 1 if t.shape[1] > 0 else 0
                t = np.hstack((t, np.array([[0], [1], [2]]) + diff))
                ix1.append(itr)
                ix2.append(jtr)
    return (
        MeshTri(p, t),
        np.array(ix1, dtype=np.int64),
        np.array(ix2, dtype=np.int64),
    )


def intersect1d(m1, m2):
    """One-dimensional supermesh."""
    p1, t1 = m1
    p2, t2 = m2
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


def elementwise_quadrature(mesh, supermesh=None, tind=None, order=None):
    """For creating element-by-element quadrature rules."""
    if order is None:
        order = 4
    if supermesh is None:
        supermesh = mesh
    X, W = get_quadrature(supermesh.elem, order)
    mmap = mesh.mapping()
    smap = supermesh.mapping()
    return (
        mmap.invF(smap.F(X), tind=tind),
        np.abs(smap.detDF(X) / mmap.detDF(X, tind=tind)) * W,
    )
