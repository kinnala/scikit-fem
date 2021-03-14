from dataclasses import replace

import numpy as np
from numpy import ndarray

from ..mesh import Mesh2D
from .mapping import Mapping


class MappingMortar(Mapping):
    """A mapping from d-1 dimensional reference element to mortar facets."""

    side = 0

    def __init__(self,
                 maps,
                 helper_to_orig,
                 helper_maps,
                 super_to_helper,
                 supermap,
                 normals):
        """Should not be called directly."""
        self.maps = maps
        self.helper_to_orig = helper_to_orig
        self.helper_maps = helper_maps
        self.super_to_helper = super_to_helper
        self.supermap = supermap
        self._normals = normals

    @classmethod
    def init_2D(cls,
                mesh1: Mesh2D,
                mesh2: Mesh2D,
                boundary1: ndarray,
                boundary2: ndarray,
                tangent: ndarray):
        """Create mortar mappings for two 2D meshes via projection.

        Parameters
        ----------
        mesh1
            An object of the type :class:`~skfem.mesh.mesh_2d.Mesh2D`.
        mesh2
            An object of the type :class:`~skfem.mesh.mesh_2d.Mesh2D`.
        boundary1
            A subset of facets to use from mesh1.
        boundary2
            A subset of facets to use from mesh2.
        tangent
            A tangent vector defining the direction of the projection.

        """
        from ..mesh import MeshLine
        tangent /= np.linalg.norm(tangent)

        # find unique nodes on the two boundaries
        p1_ix = np.unique(mesh1.facets[:, boundary1].flatten())
        p2_ix = np.unique(mesh2.facets[:, boundary2].flatten())
        p1 = mesh1.p[:, p1_ix]
        p2 = mesh2.p[:, p2_ix]

        def proj(p):
            """Project onto the line defined by 'tangent'."""
            return np.outer(tangent, tangent) @ p

        def param(p):
            """Calculate signed distances of projected points from origin."""
            y = proj(p)
            return np.linalg.norm(y, axis=0) * np.sign(np.dot(tangent, y))

        # find unique supermesh facets by combining nodes from both sides
        param_p1 = param(p1)
        param_p2 = param(p2)
        _, ix = np.unique(np.concatenate((param_p1, param_p2)),
                          return_index=True)
        ixorig = np.concatenate((p1_ix, p2_ix + mesh1.p.shape[1]))[ix]
        p = np.array([np.hstack((param(mesh1.p), param(mesh2.p)))])
        t = np.array([ixorig[:-1], ixorig[1:]])

        # create 1-dimensional supermesh from the intersections of the
        # projected facet elements
        p = p[:, np.concatenate((t[0], np.array([t[1, -1]])))]
        range_max = np.min([np.max(param_p1), np.max(param_p2)])
        range_min = np.max([np.min(param_p1), np.min(param_p2)])
        p = np.array([p[0, (p[0] <= range_max) * (p[0] >= range_min)]])
        t = np.array([np.arange(p.shape[1] - 1), np.arange(1, p.shape[1])])
        m_super = MeshLine(p, t)

        # helper meshes for creating the mappings
        m1 = MeshLine(np.sort(param_p1), np.array([np.arange(p1.shape[1] - 1),
                                                   np.arange(1, p1.shape[1])]))
        m2 = MeshLine(np.sort(param_p2), np.array([np.arange(p2.shape[1] - 1),
                                                   np.arange(1, p2.shape[1])]))

        # construct normals by rotating 'tangent'
        normal = np.array([tangent[1], -tangent[0]])
        normals = normal[:, None].repeat(t.shape[1], axis=1)

        # initialize mappings (for orienting)
        map_super = m_super._mapping()
        map_m1 = m1._mapping()
        map_m2 = m2._mapping()
        map_mesh1 = mesh1._mapping()
        map_mesh2 = mesh2._mapping()

        # matching of elements in the supermesh and the helper meshes
        mps = map_super.F(np.array([[.5]]))
        ix1 = np.digitize(mps[0, :, 0], m1.p[0]) - 1
        ix2 = np.digitize(mps[0, :, 0], m2.p[0]) - 1

        # for each element, map two points to global coordinates, reparametrize
        # the points, and flip corresponding helper mesh element indices if
        # sorting is wrong
        f1mps = .5 * (mesh1.p[:, mesh1.facets[0, boundary1]] +
                      mesh1.p[:, mesh1.facets[1, boundary1]])
        sort_boundary1 = np.argsort(param(f1mps))
        z1 = map_mesh1.G(map_m1.invF(map_super.F(np.array([[.25, .75]])),
                                     tind=ix1),
                         find=boundary1[sort_boundary1][ix1])
        ix1_flip = np.unique(ix1[param(z1[:, :, 1]) < param(z1[:, :, 0])])
        m1t = m1.t.copy()
        m1t[:, ix1_flip] = np.flipud(m1t[:, ix1_flip])
        m1 = replace(m1, t=m1t)

        f2mps = .5 * (mesh2.p[:, mesh2.facets[0, boundary2]] +
                      mesh2.p[:, mesh2.facets[1, boundary2]])
        sort_boundary2 = np.argsort(param(f2mps))
        z2 = map_mesh2.G(map_m2.invF(map_super.F(np.array([[.25, .75]])),
                                     tind=ix2),
                         find=boundary2[sort_boundary2][ix2])
        ix2_flip = np.unique(ix2[param(z2[:, :, 1]) < param(z2[:, :, 0])])
        m2t = m2.t.copy()
        m2t[:, ix2_flip] = np.flipud(m2t[:, ix2_flip])
        m2 = replace(m2, t=m2t)

        # construct normals by rotating 'tangent'
        normal = np.array([tangent[1], -tangent[0]])
        normals = normal[:, None].repeat(t.shape[1], axis=1)

        # initialize mappings (for orienting)
        map_super = m_super._mapping()
        map_m1 = m1._mapping()
        map_m2 = m2._mapping()
        map_mesh1 = mesh1._mapping()
        map_mesh2 = mesh2._mapping()

        # matching of elements in the supermesh and the helper meshes
        mps = map_super.F(np.array([[.5]]))
        ix1 = np.digitize(mps[0, :, 0], m1.p[0]) - 1
        ix2 = np.digitize(mps[0, :, 0], m2.p[0]) - 1

        return cls((map_mesh1, map_mesh2),
                   (boundary1[sort_boundary1][ix1],
                    boundary2[sort_boundary2][ix2]),
                   (map_m1, map_m2),
                   (ix1, ix2),
                   map_super,
                   normals)

    def F(self, X, tind=None):
        return self.maps[self.side].F(X, tind=tind)

    def invF(self, x, tind=None):
        return self.maps[self.side].invF(x, tind=tind)

    def detDF(self, X, tind=None):
        return self.maps[self.side].detDF(X, tind=tind)

    def DF(self, X, tind=None):
        return self.maps[self.side].DF(X, tind=tind)

    def invDF(self, X, tind=None):
        return self.maps[self.side].invDF(X, tind=tind)

    def G(self,
          X: ndarray,
          find: ndarray = None) -> ndarray:
        side = self.side
        return self.maps[side].G(
            self.helper_maps[side].invF(self.supermap.F(X),
                                        tind=self.super_to_helper[side]),
            find=self.helper_to_orig[side]
        )

    def detDG(self,
              X: ndarray,
              find: ndarray = None) -> ndarray:
        if X.shape[0] == 1:
            segs = self.G(np.array([[0., 1.]]))
            return np.sqrt(np.diff(segs[0], axis=1) ** 2 +
                           np.diff(segs[1], axis=1) ** 2)
        raise NotImplementedError

    def normals(self,
                X: ndarray,
                tind: ndarray,
                find: ndarray,
                t2f: ndarray) -> ndarray:
        return np.repeat(self._normals[:, :, None],
                         X.shape[-1],
                         axis=2)
