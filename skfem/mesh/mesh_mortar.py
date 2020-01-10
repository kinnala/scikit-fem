import numpy as np

from .mesh import Mesh
from ..mapping import MappingAffine


class MeshMortar(Mesh):
    """An interface mesh for mortar methods."""

    name = "Mortar"

    def __init__(self, mesh1, mesh2, p, facets, f2t, normals):
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
        self.p = p
        self.facets = facets
        self.f2t = f2t
        self.normals = normals
        self.target_mesh = 2 * [None]
        self.target_mesh[0] = mesh1
        self.target_mesh[1] = mesh2

        # for quadrature rules
        self.brefdom = mesh1.brefdom

        # dummy values so that initialising Mapping doesn't crash
        self.t = mesh1.t[:, :1]
        self.t2f = -1 + 0 * self.t


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

        # find unique facets by combining nodes from both sides
        _, ix = np.unique(np.concatenate((param(p1), param(p2))), return_index=True)
        ixorig = np.concatenate((p1_ix, p2_ix + mesh1.p.shape[1]))[ix]
        p = np.hstack((mesh1.p, mesh2.p)) # TODO has duplicate nodes
        facets = np.array([ixorig[:-1], ixorig[1:]])

        # construct normals
        normal = np.array([tangent[1], -tangent[0]])
        normals = normal[:, None].repeat(facets.shape[1], axis=1)

        # mappings from facets to the original triangles
        f2t = facets * 0 - 1
        for itr in range(facets.shape[1]):
            mp = .5 * (p[:, facets[0, itr]] + p[:, facets[1, itr]])
            val = param(mp)
            for jtr in boundary1:
                x1 = mesh1.p[:, mesh1.facets[0, jtr]]
                x2 = mesh1.p[:, mesh1.facets[1, jtr]]
                if (val > param(x1) and val < param(x2) or 
                    val < param(x1) and val > param(x2) or 
                    val >= param(x1) and val < param(x2) or 
                    val > param(x1) and val <= param(x2) or 
                    val <= param(x1) and val > param(x2) or 
                    val < param(x1) and val >= param(x2)):
                    f2t[0, itr] = mesh1.f2t[0, jtr]
                    break
            for jtr in boundary2:
                x1 = mesh2.p[:, mesh2.facets[0, jtr]]
                x2 = mesh2.p[:, mesh2.facets[1, jtr]]
                if (val > param(x1) and val < param(x2) or 
                    val < param(x1) and val > param(x2) or 
                    val >= param(x1) and val < param(x2) or 
                    val > param(x1) and val <= param(x2) or 
                    val <= param(x1) and val > param(x2) or 
                    val < param(x1) and val >= param(x2)):
                    f2t[1, itr] = mesh2.f2t[0, jtr]
                    break
        if not (f2t > -1).all():
            raise Exception("All mesh facets corresponding to mortar facets not found! {}".format(f2t))

        return cls(mesh1, mesh2, p, facets, f2t, normals)

    def mapping(self):
        return MappingAffine(self)
