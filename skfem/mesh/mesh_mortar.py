import numpy as np

from .mesh import Mesh
from .mesh_line import MeshLine
from ..mapping import MappingAffine


class MeshMortar(Mesh):
    """An interface mesh for mortar methods."""

    name = "Mortar"

    def __init__(self, m_super, m1, m2, mesh1, mesh2, normals, ix1,ix2,I1,I2):
        """Create an interface mesh for mortar methods.

        Parameters
        ----------
        mesh1
        mesh2
        p
        facets
        f2t
        normals

        """
        self.p = m_super.p
        self.t = m_super.t
        self.normals = normals
        self.supermesh = m_super
        self.helper_mesh = 2 * [None]
        self.helper_mesh[0] = m1
        self.helper_mesh[1] = m2
        self.target_mesh = 2 * [None]
        self.target_mesh[0] = mesh1
        self.target_mesh[1] = mesh2
        self.ix={}
        self.ix[0]=ix1
        self.ix[1]=ix2
        self.I ={}
        self.I[0]=I1
        self.I[1]=I2


    @classmethod
    def init_1D(cls, mesh1, mesh2, boundary1, boundary2, tangent):
        """Create a mortar mesh between two 2D meshes via projection.

        Parameters
        ----------
        mesh1
        mesh2
        boundary1
            A subset of facets from mesh1.
        boundary2
            A subset of facets from mesh2.
        tangent
            A tangent vector defining the direction of the projection.

        """
        tangent /= np.linalg.norm(tangent)

        # find nodes on the two boundaries
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
            return np.linalg.norm(y, axis=0) * np.dot(tangent, y)

        param_p1 = param(p1)
        param_p2 = param(p2)

        # find unique facets by combining nodes from both sides
        _, ix = np.unique(np.concatenate((param_p1, param_p2)), return_index=True)
        ixorig = np.concatenate((p1_ix, p2_ix + mesh1.p.shape[1]))[ix]
        p = np.array([np.hstack((param(mesh1.p), param(mesh2.p)))])
        t = np.array([ixorig[:-1], ixorig[1:]])

        # supermesh
        p = p[:, np.concatenate((t[0], np.array([t[1, -1]])))]
        t = np.array([np.arange(p.shape[1] - 1), np.arange(1, p.shape[1])])
        m_super = MeshLine(p, t)

        # helper meshes
        m1 = MeshLine(np.sort(param_p1), np.array([np.arange(p1.shape[1] - 1),
                                                   np.arange(1, p1.shape[1])]))
        m2 = MeshLine(np.sort(param_p2), np.array([np.arange(p2.shape[1] - 1),
                                                   np.arange(1, p2.shape[1])]))

        # construct normals by rotating 'tangent'
        normal = np.array([tangent[1], -tangent[0]])
        normals = normal[:, None].repeat(t.shape[1], axis=1)

        # initialize mappings for orienting
        map_super = m_super.mapping()
        map_m1 = m1.mapping()
        map_m2 = m2.mapping()
        map_mesh1 = mesh1.mapping()
        map_mesh2 = mesh2.mapping()

        # orient helper meshes
        mps = map_super.F(np.array([[0.5]]))
        ix1 = np.digitize(mps[0,:,0], m1.p[0]) - 1
        ix2 = np.digitize(mps[0,:,0], m2.p[0]) - 1

        # for each element, map two points to global coordinates, reparametrize
        # the points, and flip corresponding helper mesh element indices if
        # sorting is wrong
        f1mps = .5 * (mesh1.p[:, mesh1.facets[0, boundary1]] +
                      mesh1.p[:, mesh1.facets[1, boundary1]])
        sort_boundary1 = np.argsort(param(f1mps))
        z1 = map_mesh1.G(map_m1.invF(map_super.F(np.array([[0.25, 0.75]])), tind=ix1),
                         find=boundary1[sort_boundary1][ix1])
        ix1_flip = np.unique(ix1[param(z1[:,:,1]) < param(z1[:,:,0])])
        m1.t[:, ix1_flip] = np.flipud(m1.t[:, ix1_flip])

        f2mps = .5 * (mesh2.p[:, mesh2.facets[0, boundary2]] +
                      mesh2.p[:, mesh2.facets[1, boundary2]])
        sort_boundary2 = np.argsort(param(f2mps))
        z2 = map_mesh2.G(map_m2.invF(map_super.F(np.array([[0.25, 0.75]])), tind=ix2),
                         find=boundary2[sort_boundary2][ix2])
        ix2_flip = np.unique(ix2[param(z2[:,:,1]) < param(z2[:,:,0])])
        m2.t[:, ix2_flip] = np.flipud(m2.t[:, ix2_flip])

        return cls(m_super,
                   m1, m2,
                   mesh1, mesh2,
                   normals,
                   ix1, ix2,
                   boundary1[sort_boundary1][ix1],
                   boundary2[sort_boundary2][ix2])
