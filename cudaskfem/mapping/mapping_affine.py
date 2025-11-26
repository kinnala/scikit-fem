import numpy as np

from .mapping import Mapping


class MappingAffine(Mapping):
    """An affine mapping for simplical elements."""

    def __init__(self, mesh, tind=None):
        """Local-to-global mappings for simplex elements.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.Mesh`.
        tind
            An optional array of element indices for memory
            optimizations.  If provided, only the included element
            indices are used and all methods will ignore the
            respective tind parameter.

        """
        self.dim = mesh.p.shape[0]
        self.mesh = mesh
        self.tind = tind

    @property
    def A(self):
        """Affine mapping matrix."""
        if not hasattr(self, '_A'):
            self._init_Ab()
        return self._A

    @property
    def b(self):
        """Affine mapping constant."""
        if not hasattr(self, '_b'):
            self._init_Ab()
        return self._b

    @property
    def detA(self):
        """Affine mapping determinant."""
        if not hasattr(self, '_detA'):
            self._init_invA()
        return self._detA

    @property
    def invA(self):
        """Affine mapping matrix inverse."""
        if not hasattr(self, '_invA'):
            self._init_invA()
        return self._invA

    def _init_Ab(self):
        """For lazy evaluation of interior mapping."""
        if self.mesh.t.shape[0] > 0:  # todo: why this guard exists?
            nt = self.mesh.t.shape[1] if self.tind is None else len(self.tind)
            dim = self.dim
            # initialize the affine mapping
            self._A = np.empty((dim, dim, nt))
            self._b = np.empty((dim, nt))

            if self.tind is None:
                for i in range(dim):
                    self._b[i] = self.mesh.p[i, self.mesh.t[0]]
                    for j in range(dim):
                        self._A[i, j] = (
                            self.mesh.p[i, self.mesh.t[j + 1]] -
                            self.mesh.p[i, self.mesh.t[0]]
                        )
            else:
                for i in range(dim):
                    self._b[i] = self.mesh.p[i, self.mesh.t[0, self.tind]]
                    for j in range(dim):
                        self._A[i, j] = (
                            self.mesh.p[i, self.mesh.t[j + 1, self.tind]] -
                            self.mesh.p[i, self.mesh.t[0, self.tind]]
                        )

    def _init_invA(self):
        """For lazy evaluation of interior mapping."""
        if self.mesh.t.shape[0] > 0:  # todo: why this guard exists?
            nt = self.mesh.t.shape[1] if self.tind is None else len(self.tind)
            dim = self.dim
            # determinants
            if dim == 1:
                self._detA = self.A[0, 0]
            elif dim == 2:
                self._detA = (self.A[0, 0] * self.A[1, 1] -
                              self.A[0, 1] * self.A[1, 0])
            elif dim == 3:
                self._detA = (self.A[0, 0] * (self.A[1, 1] * self.A[2, 2] -
                                              self.A[1, 2] * self.A[2, 1]) -
                              self.A[0, 1] * (self.A[1, 0] * self.A[2, 2] -
                                              self.A[1, 2] * self.A[2, 0]) +
                              self.A[0, 2] * (self.A[1, 0] * self.A[2, 1] -
                                              self.A[1, 1] * self.A[2, 0]))
            else:
                raise Exception("Not implemented for the given dimension.")

            # affine mapping inverses
            self._invA = np.empty((dim, dim, nt))
            if dim == 1:
                self._invA[0, 0] = 1. / self.A[0, 0]
            elif dim == 2:
                self._invA[0, 0] =  self.A[1, 1] / self.detA  # noqa
                self._invA[0, 1] = -self.A[0, 1] / self.detA
                self._invA[1, 0] = -self.A[1, 0] / self.detA
                self._invA[1, 1] =  self.A[0, 0] / self.detA  # noqa
            elif dim == 3:
                self._invA[0, 0] = (-self.A[1, 2] * self.A[2, 1] +
                                    self.A[1, 1] * self.A[2, 2]) / self.detA
                self._invA[1, 0] = (self.A[1, 2] * self.A[2, 0] -
                                    self.A[1, 0] * self.A[2, 2]) / self.detA
                self._invA[2, 0] = (-self.A[1, 1] * self.A[2, 0] +
                                    self.A[1, 0] * self.A[2, 1]) / self.detA
                self._invA[0, 1] = (self.A[0, 2] * self.A[2, 1] -
                                    self.A[0, 1] * self.A[2, 2]) / self.detA
                self._invA[1, 1] = (-self.A[0, 2] * self.A[2, 0] +
                                    self.A[0, 0] * self.A[2, 2]) / self.detA
                self._invA[2, 1] = (self.A[0, 1] * self.A[2, 0] -
                                    self.A[0, 0] * self.A[2, 1]) / self.detA
                self._invA[0, 2] = (-self.A[0, 2] * self.A[1, 1] +
                                    self.A[0, 1] * self.A[1, 2]) / self.detA
                self._invA[1, 2] = (self.A[0, 2] * self.A[1, 0] -
                                    self.A[0, 0] * self.A[1, 2]) / self.detA
                self._invA[2, 2] = (-self.A[0, 1] * self.A[1, 0] +
                                    self.A[0, 0] * self.A[1, 1]) / self.detA
            else:
                raise Exception("Not implemented for the given dimension.")

    @property
    def B(self):
        """Boundary affine mapping matrix."""
        if not hasattr(self, '_B'):
            self._init_boundary_mapping()
        return self._B

    @property
    def c(self):
        """Boundary affine mapping constant."""
        if not hasattr(self, '_c'):
            self._init_boundary_mapping()
        return self._c

    @property
    def detB(self):
        """Boundary affine mapping determinant."""
        if not hasattr(self, '_detB'):
            self._init_boundary_mapping()
        return self._detB

    def _init_boundary_mapping(self):
        """For lazy evaluation of boundary mapping."""
        dim = self.dim
        nf = self.mesh.facets.shape[1]
        # initialize the boundary mapping
        self._B = np.empty((dim, dim - 1, nf))
        self._c = np.empty((dim, nf))

        for i in range(dim):
            self._c[i] = self.mesh.p[i, self.mesh.facets[0]]
            for j in range(dim - 1):
                self._B[i, j] = (self.mesh.p[i, self.mesh.facets[j + 1]] -
                                 self.mesh.p[i, self.mesh.facets[0]])

        # area scaling
        if dim == 1:
            self._detB = np.ones(nf)
        elif dim == 2:
            self._detB = np.sqrt(self._B[0, 0] ** 2 + self._B[1, 0] ** 2)
        elif dim == 3:
            self._detB = np.sqrt((self._B[1, 0] * self._B[2, 1] -
                                  self._B[2, 0] * self._B[1, 1]) ** 2 +
                                 (-self._B[0, 0] * self._B[2, 1] +
                                  self._B[2, 0] * self._B[0, 1]) ** 2 +
                                 (self._B[0, 0] * self._B[1, 1] -
                                  self._B[1, 0] * self._B[0, 1]) ** 2)
        else:
            raise Exception("Not implemented for the given dimension.")

    def F(self, X, tind=None):
        if tind is None or self.tind is not None:
            A, b = self.A, self.b
        else:
            A, b = self.A[:, :, tind], self.b[:, tind]

        if len(X.shape) == 2:
            return (np.einsum('ijk,jl', A, X).T + b.T).T
        elif len(X.shape) == 3:
            return (np.einsum('ijk,jkl->ikl', A, X).T + b.T).T
        raise NotImplementedError

    def invF(self, x, tind=None):
        if tind is None or self.tind is not None:
            invA, b = self.invA, self.b
        else:
            invA, b = self.invA[:, :, tind], self.b[:, tind]

        y = (x.T - b.T).T

        return np.einsum('ijk,jkl->ikl', invA, y)

    def detDF(self, X, tind=None):
        if tind is None or self.tind is not None:
            detDF = self.detA
        else:
            detDF = self.detA[tind]

        return np.tile(detDF, (X.shape[-1], 1)).T

    def DF(self, X, tind=None):
        if tind is None or self.tind is not None:
            DF = self.A
        else:
            DF = self.A[:, :, tind]

        if len(X.shape) == 2:
            return np.einsum('ijk,l->ijkl', DF, 1 + np.zeros_like(X[0]))
        elif len(X.shape) == 3:
            return np.einsum('ijk,kl->ijkl', DF, 1 + np.zeros_like(X[0]))
        raise NotImplementedError

    def invDF(self, X, tind=None):
        if tind is None or self.tind is not None:
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

        if len(X.shape) == 2:
            return (np.einsum('ijk,jl', B, X).T + c.T).T
        elif len(X.shape) == 3:
            return (np.einsum('ijk,jkl->ikl', B, X).T + c.T).T

        raise Exception("Wrong dimension of input.")

    def detDG(self, X, find=None):
        if find is None:
            detDG = self.detB
        else:
            detDG = self.detB[find]

        return np.tile(detDG, (X.shape[-1], 1)).T

    def normals(self, X, tind, find, t2f):
        if self.dim == 1:
            Nref = np.array([[-1.],
                             [1.]])
        elif self.dim == 2:
            Nref = np.array([[0., -1.],
                             [1., 1.],
                             [-1., 0.]])
        elif self.dim == 3:
            Nref = np.array([[0., 0., -1.],
                             [0., -1., 0.],
                             [-1., 0., 0.],
                             [1., 1., 1.]])
        else:
            raise Exception("Not implemented for the given dimension.")

        invDF = self.invDF(X, tind)
        N = np.empty((self.dim, len(find)))

        for itr in range(Nref.shape[0]):
            ix = np.nonzero(t2f[itr, tind] == find)[0].astype(np.int32)
            for jtr in range(Nref.shape[1]):
                N[jtr, ix] = Nref[itr, jtr]

        n = np.einsum('ijkl,ik->jkl', invDF, N)
        nlength = np.sqrt(np.sum(n ** 2, axis=0))
        return np.einsum('ijk,jk->ijk', n, 1. / nlength)
