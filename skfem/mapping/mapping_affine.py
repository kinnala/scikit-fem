import numpy as np

from .mapping import Mapping


class MappingAffine(Mapping):
    """An affine mapping for simplical elements."""
    def __init__(self, mesh):
        dim = mesh.p.shape[0]

        if mesh.t.shape[0] > 0:
            nt = mesh.t.shape[1]
            # initialize the affine mapping
            self.A = np.empty((dim, dim, nt))
            self.b = np.empty((dim, nt))

            for i in range(dim):
                self.b[i] = mesh.p[i, mesh.t[0, :]]
                for j in range(dim):
                    self.A[i, j] = (mesh.p[i, mesh.t[j + 1, :]] -
                                    mesh.p[i, mesh.t[0, :]])

            # determinants
            if dim == 1:
                self.detA = self.A[0, 0]
            elif dim == 2:
                self.detA = (self.A[0, 0] * self.A[1, 1] -
                             self.A[0, 1] * self.A[1, 0])
            elif dim == 3:
                self.detA = self.A[0, 0] * (self.A[1, 1] * self.A[2, 2] -
                                            self.A[1, 2] * self.A[2, 1]) -\
                            self.A[0, 1] * (self.A[1, 0] * self.A[2, 2] -
                                            self.A[1, 2] * self.A[2, 0]) +\
                            self.A[0, 2] * (self.A[1, 0] * self.A[2, 1] -
                                            self.A[1, 1] * self.A[2, 0])
            else:
                raise Exception("Not implemented for the given dimension.")

            # affine mapping inverses
            self.invA = np.empty((dim, dim, nt))
            if dim == 1:
                self.invA[0, 0] = 1.0 / self.A[0, 0]
            elif dim == 2:
                self.invA[0, 0] =  self.A[1, 1] / self.detA
                self.invA[0, 1] = -self.A[0, 1] / self.detA
                self.invA[1, 0] = -self.A[1, 0] / self.detA
                self.invA[1, 1] =  self.A[0, 0] / self.detA
            elif dim == 3:
                self.invA[0, 0] = (-self.A[1, 2] * self.A[2, 1] +
                                   self.A[1, 1] * self.A[2, 2]) / self.detA
                self.invA[1, 0] = (self.A[1, 2] * self.A[2, 0] -
                                   self.A[1, 0] * self.A[2, 2]) / self.detA
                self.invA[2, 0] = (-self.A[1, 1] * self.A[2, 0] +
                                   self.A[1, 0] * self.A[2, 1]) / self.detA
                self.invA[0, 1] = (self.A[0, 2] * self.A[2, 1] -
                                   self.A[0, 1] * self.A[2, 2]) / self.detA
                self.invA[1, 1] = (-self.A[0, 2] * self.A[2, 0] +
                                   self.A[0, 0] * self.A[2, 2]) / self.detA
                self.invA[2, 1] = (self.A[0, 1] * self.A[2, 0] -
                                   self.A[0, 0] * self.A[2, 1]) / self.detA
                self.invA[0, 2] = (-self.A[0, 2] * self.A[1, 1] +
                                   self.A[0, 1] * self.A[1, 2]) / self.detA
                self.invA[1, 2] = (self.A[0, 2] * self.A[1, 0] -
                                   self.A[0, 0] * self.A[1, 2]) / self.detA
                self.invA[2, 2] = (-self.A[0, 1] * self.A[1, 0] +
                                   self.A[0, 0] * self.A[1, 1]) / self.detA
            else:
                raise Exception("Not implemented for the given dimension.")

        if hasattr(mesh, 'facets'):
            nf = mesh.facets.shape[1]
            # initialize the boundary mapping
            self.B = np.empty((dim, dim - 1, nf))
            self.c = np.empty((dim, nf))

            for i in range(dim):
                self.c[i] = mesh.p[i, mesh.facets[0, :]]
                for j in range(dim-1):
                    self.B[i, j] = (mesh.p[i, mesh.facets[j + 1, :]] -
                                    mesh.p[i, mesh.facets[0, :]])

            # area scaling
            if dim == 1:
                self.detB = np.ones(nf)
            elif dim == 2:
                self.detB = np.sqrt(self.B[0, 0]**2 + self.B[1, 0]**2)
            elif dim == 3:
                self.detB = np.sqrt((self.B[1, 0]*self.B[2, 1] -
                                     self.B[2, 0]*self.B[1, 1])**2 +
                                    (-self.B[0, 0]*self.B[2, 1] +
                                     self.B[2, 0]*self.B[0, 1])**2 +
                                    (self.B[0, 0]*self.B[1, 1] -
                                     self.B[1, 0]*self.B[0, 1])**2)
            else:
                raise Exception("Not implemented for the given dimension.")

        self.dim = dim
        self.mesh = mesh  # this is required in ElementH2

    def F(self, X, tind=None):
        if tind is None:
            A, b = self.A, self.b
        else:
            A, b = self.A[:, :, tind], self.b[:, tind]

        if len(X.shape) == 2:
            return (np.einsum('ijk,jl', A, X).T + b.T).T
        elif len(X.shape) == 3:
            return (np.einsum('ijk,jkl->ikl', A, X).T + b.T).T
        else:
            raise Exception("Wrong dimension of input.")

    def invF(self, x, tind=None):
        if tind is None:
            invA, b = self.invA, self.b
        else:
            invA, b = self.invA[:, :, tind], self.b[:, tind]

        y = (x.T - b.T).T

        return np.einsum('ijk,jkl->ikl', invA, y)

    def detDF(self, X, tind=None):
        if tind is None:
            detDF = self.detA
        else:
            detDF = self.detA[tind]

        return np.tile(detDF, (X.shape[1], 1)).T

    def DF(self, X, tind=None):
        if tind is None:
            DF = self.A
        else:
            DF = self.A[:, :, tind]

        return np.einsum('ijk,l->ijkl', DF, 1 + 0*X[0, :])

    def invDF(self, X, tind=None):
        if tind is None:
            invDF = self.invA
        else:
            invDF = self.invA[:, :, tind]

        ones = np.ones(X.shape[-1])
        return np.einsum('ijk,l->ijkl', invDF, ones)

    def G(self, X, find=None):
        if find is None:
            B, c = self.B, self.c
        else:
            B, c = self.B[:, :, find], self.c[:, find]

        return (np.einsum('ijk,jl', B, X).T + c.T).T

    def detDG(self, X, find=None):
        if find is None:
            detDG = self.detB
        else:
            detDG = self.detB[find]

        return np.tile(detDG, (X.shape[1], 1)).T

    def normals(self, X, tind, find, t2f):
        if self.dim == 1:
            Nref = np.array([[-1.0],
                             [ 1.0]])
        elif self.dim == 2:
            Nref = np.array([[0.0, -1.0],
                             [1.0, 1.0],
                             [-1.0, 0.0]])
        elif self.dim == 3:
            Nref = np.array([[0.0, 0.0, -1.0],
                             [0.0, -1.0, 0.0],
                             [-1.0, 0.0, 0.0],
                             [1.0, 1.0, 1.0]])
        else:
            raise Exception("Not implemented for the given dimension.")

        invDF = self.invDF(X, tind)
        N = np.empty((self.dim, len(find)))

        for itr in range(Nref.shape[0]):
            ix = np.nonzero(t2f[itr, tind] == find)[0]
            for jtr in range(Nref.shape[1]):
                N[jtr, ix] = Nref[itr, jtr]

        n = np.einsum('ijkl,ik->jkl', invDF, N)
        nlength = np.sqrt(np.sum(n**2, axis=0))
        return np.einsum('ijk,jk->ijk', n, 1.0/nlength)
