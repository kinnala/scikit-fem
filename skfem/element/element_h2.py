import numpy as np
from .element import Element


class ElementH2(Element):
    """Elements defined implicitly through global degrees-of-freedom."""
    
    order = (0, 1, 2)
    V = None  # For caching inverse Vandermonde matrix

    def gbasis(self, mapping, X, i, tind=None):
        if tind is None:
            tind = np.arange(mapping.mesh.t.shape[1])
        # initialize power basis
        self._pbasis_init(self.maxdeg)
        N = len(self._pbasis)

        if self.V is None:
            # construct Vandermonde matrix and invert it
            self.V = np.linalg.inv(self._eval_dofs(mapping.mesh))

        V = self.V[tind]

        x = mapping.F(X, tind=tind)
        u = np.zeros(x[0].shape)
        du = np.zeros((2,) + x[0].shape)
        ddu = np.zeros((2, 2) + x[0].shape)

        # loop over new basis
        for itr in range(N):
            u += V[:, itr, i][:, None]\
                 * self._pbasis[itr](x[0], x[1])
            du[0] += V[:, itr, i][:, None]\
                     * self._pbasisdx[itr](x[0], x[1])
            du[1] += V[:, itr, i][:,None]\
                     * self._pbasisdy[itr](x[0], x[1])
            ddu[0, 0] += V[:, itr, i][:, None]\
                         * self._pbasisdxx[itr](x[0], x[1])
            ddu[0, 1] += V[:, itr, i][:, None]\
                         * self._pbasisdxy[itr](x[0], x[1])
            ddu[1, 1] += V[:, itr, i][:, None]\
                         * self._pbasisdyy[itr](x[0], x[1])

        # dxy = dyx
        ddu[1, 0] = ddu[0, 1]

        return u, du, ddu

    def _pbasis_create_xy(self, i, j, dx=0, dy=0):
        cx = 1
        cy = 1
        if dx > 0:
            for k in np.arange(dx, 0, -1):
                cx *= i - dx + k
        if dy > 0:
            for k in np.arange(dy, 0, -1):
                cy *= j - dy + k
        return eval("lambda x, y: {}*x**{}*y**{}".format(cx * cy,
                                                         np.max([i - dx, 0]),
                                                         np.max([j - dy, 0])))

    def _pbasis_init(self, N):
        """Define power bases (for 2D)."""
        if not hasattr(self, '_pbasis'):
            setattr(self, '_pbasis', [self._pbasis_create_xy(i, j)
                                      for i in range(N+1)
                                      for j in range(N+1)
                                      if i + j <= N])
            setattr(self, '_pbasisdx', [self._pbasis_create_xy(i, j,
                                                               dx=1)
                                        for i in range(N+1)
                                        for j in range(N+1)
                                        if i + j <= N])
            setattr(self, '_pbasisdy', [self._pbasis_create_xy(i, j,
                                                               dy=1)
                                        for i in range(N+1)
                                        for j in range(N+1)
                                        if i + j <= N])
            setattr(self, '_pbasisdxx', [self._pbasis_create_xy(i, j,
                                                                dx=2)
                                         for i in range(N+1)
                                         for j in range(N+1)
                                         if i + j <= N])
            setattr(self, '_pbasisdxy', [self._pbasis_create_xy(i, j,
                                                                dx=1,
                                                                dy=1)
                                         for i in range(N+1)
                                         for j in range(N+1)
                                         if i + j <= N])
            setattr(self, '_pbasisdyy', [self._pbasis_create_xy(i, j,
                                                                dy=2)
                                         for i in range(N+1)
                                         for j in range(N+1)
                                         if i + j <= N])

    def _eval_dofs(self, mesh, tind=None):
        if tind is None:
            tind = np.arange(mesh.t.shape[1])
        N = len(self._pbasis)

        V = np.zeros((len(tind), N, N))

        if mesh.t.shape[0] == 3:
            # vertices, edges, tangents, normals
            v = np.empty((3, 2, len(tind)))
            e = np.empty((3, 2, len(tind)))
            n = np.empty((3, 2, len(tind)))

            # vertices
            for itr in range(3):
                v[itr] = mesh.p[:, mesh.t[itr, tind]]

            # edge midpoints
            e[0] = 0.5 * (v[0] + v[1])
            e[1] = 0.5 * (v[1] + v[2])
            e[2] = 0.5 * (v[0] + v[2])

            # normal vectors
            n[0] = v[0] - v[1]
            n[1] = v[1] - v[2]
            n[2] = v[0] - v[2]

            for itr in range(3):
                n[itr] = np.array([n[itr, 1, :], -n[itr, 0, :]])
                n[itr] /= np.linalg.norm(n[itr], axis=0)
        else:
            raise NotImplementedError("The used mesh type not supported "
                                      "in ElementH2.")

        # evaluate dofs, gdof implemented in subclasses
        for itr in range(N):
            for jtr in range(N):
                u = self._pbasis[itr]
                du = [self._pbasisdx[itr], self._pbasisdy[itr]]
                ddu = [self._pbasisdxx[itr],
                       self._pbasisdxy[itr],
                       self._pbasisdyy[itr]]
                V[:, jtr, itr] = self.gdof(u, du, ddu, v, e, n, jtr)

        return V
