import numpy as np

from ..element_global import ElementGlobal
from ...refdom import RefTet


def _sorted_vertices(w, ix):
    """Return the vertices of a subentity sorted by the global index."""
    t = w['tix'][ix]
    order = np.argsort(t, axis=0)
    v = w['v'][ix]
    return np.take_along_axis(v, order[:, None, :], axis=0)


def _face(w, ix):
    """Return the centroid and the unit normal of a face.

    The normal is oriented using the global vertex indices so that it is the
    same for both tetrahedra sharing the face.

    """
    a, b, c = _sorted_vertices(w, ix)
    n = np.cross(b - a, c - a, axis=0)
    return (a + b + c) / 3., n / np.linalg.norm(n, axis=0)


def _edge(w, ix):
    """Return the endpoints, the midpoint and two unit normals of an edge.

    The normals depend only on the endpoints, ordered using the global vertex
    indices, so that they are the same for all tetrahedra sharing the edge.

    """
    a, b = _sorted_vertices(w, ix)
    t = (b - a) / np.linalg.norm(b - a, axis=0)
    # cross with the coordinate axis that is the least parallel to the edge
    e = np.zeros_like(t)
    e[np.argmin(np.abs(t), axis=0), np.arange(t.shape[1])] = 1.
    n1 = np.cross(t, e, axis=0)
    n1 /= np.linalg.norm(n1, axis=0)
    n2 = np.cross(t, n1, axis=0)
    return a, b, .5 * (a + b), n1, n2


def _dn(F, x, n):
    """Evaluate the directional derivative."""
    return sum(F[(j,)](*x) * n[j] for j in range(3))


def _dnn(F, x, n):
    """Evaluate the second order directional derivative."""
    return sum(F[(j, k)](*x) * n[j] * n[k]
               for j in range(3)
               for k in range(3))


def _simpson(f, a, b, m):
    """Mean value over an edge, exact for quadratic polynomials."""
    return (f(a) + 4. * f(m) + f(b)) / 6.


class ElementTetMorley(ElementGlobal):
    """The three-dimensional Morley element.

    The Morley-Wang-Xu element of degree two: the DOFs are the mean values
    over the edges and the mean values of the normal derivative over the
    faces.  Nonconforming for fourth order problems.

    """

    edge_dofs = 1
    facet_dofs = 1
    maxdeg = 2
    dofnames = ['u', 'u_n']
    doflocs = np.array([[.5, 0., 0.],
                        [.5, .5, 0.],
                        [0., .5, 0.],
                        [0., 0., .5],
                        [.5, 0., .5],
                        [0., .5, .5],
                        [1. / 3., 1. / 3., 0.],
                        [1. / 3., 0., 1. / 3.],
                        [0., 1. / 3., 1. / 3.],
                        [1. / 3., 1. / 3., 1. / 3.]])
    refdom = RefTet

    def gdof(self, F, w, i):
        if i < 6:
            a, b, m, _, _ = _edge(w, self.refdom.edges[i])
            return _simpson(lambda x: F[()](*x), a, b, m)
        elif i < 10:
            c, n = _face(w, self.refdom.facets[i - 6])
            return _dn(F, c, n)
        self._index_error()


class ElementTetMWX3(ElementGlobal):
    """The Morley-Wang-Xu element of degree three.

    The DOFs are the point values at the vertices, the mean values of the two
    normal derivatives over the edges, and the mean values of the second
    normal derivative over the faces.  Nonconforming for sixth order
    problems.

    """

    nodal_dofs = 1
    edge_dofs = 2
    facet_dofs = 1
    maxdeg = 3
    derivatives = 3
    dofnames = ['u', 'u_n', 'u_n', 'u_nn']
    doflocs = np.array([[0., 0., 0.],
                        [1., 0., 0.],
                        [0., 1., 0.],
                        [0., 0., 1.],
                        [.5, 0., 0.],
                        [.5, 0., 0.],
                        [.5, .5, 0.],
                        [.5, .5, 0.],
                        [0., .5, 0.],
                        [0., .5, 0.],
                        [0., 0., .5],
                        [0., 0., .5],
                        [.5, 0., .5],
                        [.5, 0., .5],
                        [0., .5, .5],
                        [0., .5, .5],
                        [1. / 3., 1. / 3., 0.],
                        [1. / 3., 0., 1. / 3.],
                        [0., 1. / 3., 1. / 3.],
                        [1. / 3., 1. / 3., 1. / 3.]])
    refdom = RefTet

    def gdof(self, F, w, i):
        if i < 4:
            return F[()](*w['v'][i])
        elif i < 16:
            a, b, m, n1, n2 = _edge(w, self.refdom.edges[(i - 4) // 2])
            n = n1 if i % 2 == 0 else n2
            return _simpson(lambda x: _dn(F, x, n), a, b, m)
        elif i < 20:
            c, n = _face(w, self.refdom.facets[i - 16])
            return _dnn(F, c, n)
        self._index_error()
