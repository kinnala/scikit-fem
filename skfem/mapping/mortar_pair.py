from typing import NamedTuple

import numpy as np
from numpy import ndarray

from ..mesh.mesh2d import Mesh2D
from .mapping import Mapping


class MortarPair(NamedTuple):
    """A pair of mappings for mortar methods.

    In mortar methods, we are enforcing interface conditions on non-matching
    meshes. To this end, we must generate matching quadrature points on both
    sides of the interface.

    :class:`~skfem.mapping.MortarPair` is a pair of
    :class:`~skfem.mapping.Mapping` objects defined so that mapping any local
    point in the reference element using both of the mappings, we get matching
    global points on the original non-matching meshes.

    Attributes
    ----------
    mapping1
        Mapping to the facets of the first mesh.
    mapping2
        Mapping to the facets of the second mesh.

    """

    mapping1: Mapping = None
    mapping2: Mapping = None

    @classmethod
    def init_2D(cls,
                mesh1: Mesh2D,
                mesh2: Mesh2D,
                boundary1: ndarray,
                boundary2: ndarray,
                tangent: ndarray = None):
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
            If not given, use first facet of boundary1 as the vector.

        """
        from ..mesh import MeshLine

        if tangent is None:
            tangent = (mesh1.p[:, mesh1.facets[0, boundary1[0]]] -
                       mesh1.p[:, mesh1.facets[1, boundary1[0]]])

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
        _, ix = np.unique(np.concatenate((param_p1, param_p2)), return_index=True)
        ixorig = np.concatenate((p1_ix, p2_ix + mesh1.p.shape[1]))[ix]
        p = np.array([np.hstack((param(mesh1.p), param(mesh2.p)))])
        t = np.array([ixorig[:-1], ixorig[1:]])

        # create 1-dimensional supermesh from the intersections of the projected
        # facet elements
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
        map_super = m_super.mapping()
        map_m1 = m1.mapping()
        map_m2 = m2.mapping()
        map_mesh1 = mesh1.mapping()
        map_mesh2 = mesh2.mapping()

        # matching of elements in the supermesh and the helper meshes
        mps = map_super.F(np.array([[0.5]]))
        ix1 = np.digitize(mps[0, :, 0], m1.p[0]) - 1
        ix2 = np.digitize(mps[0, :, 0], m2.p[0]) - 1

        # for each element, map two points to global coordinates, reparametrize
        # the points, and flip corresponding helper mesh element indices if
        # sorting is wrong
        f1mps = .5 * (mesh1.p[:, mesh1.facets[0, boundary1]] +
                      mesh1.p[:, mesh1.facets[1, boundary1]])
        sort_boundary1 = np.argsort(param(f1mps))
        z1 = map_mesh1.G(map_m1.invF(map_super.F(np.array([[0.25, 0.75]])), tind=ix1),
                         find=boundary1[sort_boundary1][ix1])
        ix1_flip = np.unique(ix1[param(z1[:, :, 1]) < param(z1[:, :, 0])])
        m1.t[:, ix1_flip] = np.flipud(m1.t[:, ix1_flip])

        f2mps = .5 * (mesh2.p[:, mesh2.facets[0, boundary2]] +
                      mesh2.p[:, mesh2.facets[1, boundary2]])
        sort_boundary2 = np.argsort(param(f2mps))
        z2 = map_mesh2.G(map_m2.invF(map_super.F(np.array([[0.25, 0.75]])), tind=ix2),
                         find=boundary2[sort_boundary2][ix2])
        ix2_flip = np.unique(ix2[param(z2[:, :, 1]) < param(z2[:, :, 0])])
        m2.t[:, ix2_flip] = np.flipud(m2.t[:, ix2_flip])

        # create new mapping objects with G replaced by the supermesh mapping
        new_map1 = mesh1.mapping()
        new_map2 = mesh2.mapping()
        map_m1 = m1.mapping()
        map_m2 = m2.mapping()
        new_map1.G = lambda X, **_: map_mesh1.G(map_m1.invF(map_super.F(X), tind=ix1),
                                                find=boundary1[sort_boundary1][ix1])
        new_map2.G = lambda X, **_: map_mesh2.G(map_m2.invF(map_super.F(X), tind=ix2),
                                                find=boundary2[sort_boundary2][ix2])

        # these are used by :class:`skfem.assembly.basis.MortarBasis`
        new_map1.find = boundary1[sort_boundary1][ix1]
        new_map2.find = boundary2[sort_boundary2][ix2]
        new_map1.normals = normals
        new_map2.normals = normals

        # method for calculating the lengths of the segments ('detDG')
        segments1 = new_map1.G(np.array([[0.0, 1.0]]))
        segments2 = new_map2.G(np.array([[0.0, 1.0]]))
        new_map1.detDG = lambda *_, **__: np.sqrt(np.diff(segments1[0], axis=1)**2 +
                                                  np.diff(segments1[1], axis=1)**2)
        new_map2.detDG = lambda *_, **__: np.sqrt(np.diff(segments2[0], axis=1)**2 +
                                                  np.diff(segments2[1], axis=1)**2)

        return cls(mapping1=new_map1, mapping2=new_map2)
