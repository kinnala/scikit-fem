import numpy as np

from .mesh import MeshLine


def intersect(p1, t1, p2, t2):
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

    return p, t, ix1, ix2
