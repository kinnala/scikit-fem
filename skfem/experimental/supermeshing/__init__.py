import numpy as np

from skfem.mesh import MeshLine
from skfem.quadrature import get_quadrature


def intersect(m1, m2):
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

    return {
        'doflocs': p,
        't': t,
        'cell_data': {
            't1': ix1,
            't2': ix2,
        }
    }


def elementwise_quadrature(mesh, supermesh=None, key=None, order=None):
    """For creating element-by-element quadrature rules."""
    if order is None:
        order = 4
    if supermesh is None:
        supermesh = mesh
        tind = None
    else:
        tind = supermesh.cell_data[key]
    X, W = get_quadrature(supermesh.elem, order)
    mmap = mesh.mapping()
    smap = supermesh.mapping()
    return (
        mmap.invF(smap.F(X), tind=tind),
        np.abs(smap.detDF(X) / mmap.detDF(X, tind=tind)) * W,
    )
