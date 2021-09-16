import itertools

import numpy as np

from .element import Element
from .discrete_field import DiscreteField


class ElementGlobal(Element):
    """Elements defined implicitly through global degrees-of-freedom."""

    V = None  # For caching inverse Vandermonde matrix
    derivatives = 2  # By default, include first and second derivatives
    tensorial_basis = False

    def gbasis(self, mapping, X, i, tind=None):

        if tind is None:
            tind = np.arange(mapping.mesh.t.shape[1])

        if self.V is None:
            # initialize power basis
            self._pbasis_init(self.maxdeg,
                              self.dim,
                              self.derivatives,
                              self.tensorial_basis)
            # construct Vandermonde matrix and invert it
            self.V = np.linalg.inv(self._eval_dofs(mapping.mesh))

        V = self.V[tind]

        x = mapping.F(X, tind=tind)
        U = [np.zeros((self.dim,) * k + x[0].shape)
             for k in range(self.derivatives + 1)]

        N = len(self._pbasis[()])
        # loop over new basis
        for k in range(self.derivatives + 1):
            diffs = list(itertools.product(*((list(range(self.dim)),) * k)))
            for itr in range(N):
                for diff in diffs:
                    U[k][diff] += (V[:, itr, i][:, None]
                                   * self._pbasis[diff][itr](*x))

        hod = {}
        for k in range(self.derivatives - 2):
            hod['grad{}'.format(k + 3)] = U[k + 3]

        return (
            DiscreteField(
                value=U[0],
                grad=U[1],
                hess=U[2],
                **hod
            ),
        )

    def _pbasis_create(self, i, j=None, k=None, dx=0, dy=0, dz=0):
        """Return a single power basis function."""
        if j is None and k is None:  # 1d
            cx = 1
            if dx > 0:
                for l in np.arange(dx, 0, -1):
                    cx *= i - dx + l
            return eval(("lambda x: {}*x**{}"
                         .format(cx, np.max([i - dx, 0]))))
        elif k is None:  # 2d
            cx = 1
            cy = 1
            if dx > 0:
                for l in np.arange(dx, 0, -1):
                    cx *= i - dx + l
            if dy > 0:
                for l in np.arange(dy, 0, -1):
                    cy *= j - dy + l
            return eval(("lambda x, y: {}*x**{}*y**{}"
                         .format(cx * cy,
                                 np.max([i - dx, 0]),
                                 np.max([j - dy, 0]))))
        else:  # 3d
            cx = 1
            cy = 1
            cz = 1
            if dx > 0:
                for l in np.arange(dx, 0, -1):
                    cx *= i - dx + l
            if dy > 0:
                for l in np.arange(dy, 0, -1):
                    cy *= j - dy + l
            if dz > 0:
                for l in np.arange(dz, 0, -1):
                    cz *= k - dz + l
            return eval(("lambda x, y, z: {}*x**{}*y**{}*z**{}"
                         .format(cx * cy * cz,
                                 np.max([i - dx, 0]),
                                 np.max([j - dy, 0]),
                                 np.max([k - dz, 0]),)))

    def _pbasis_init(self, maxdeg, dim, Ndiff, is_tensorial=False):
        """Define power bases.

        Parameters
        ----------
        maxdeg
            Maximum degree of the basis
        dim
            Dimension of the domain.x
        Ndiff
            Number of derivatives to include.

        """
        if is_tensorial:
            maxdeg = int(maxdeg / 2)
        self._pbasis = {}
        for k in range(Ndiff + 1):
            diffs = list(itertools.product(*((list(range(dim)),) * k)))
            for diff in diffs:
                #  desc = ''.join([str(d) for d in diff])
                dx = sum([1 for d in diff if d == 0])
                dy = sum([1 for d in diff if d == 1]) if dim == 2 else None
                dz = sum([1 for d in diff if d == 2]) if dim == 3 else None
                if dim == 1:
                    self._pbasis[diff] = [
                        self._pbasis_create(i=i, dx=dx)
                        for i in range(maxdeg + 1)
                        if i <= maxdeg
                    ]
                elif dim == 2:
                    self._pbasis[diff] = [
                        self._pbasis_create(i=i, j=j, dx=dx, dy=dy)
                        for i in range(maxdeg + 1)
                        for j in range(maxdeg + 1)
                        if is_tensorial or i + j <= maxdeg
                    ]
                elif dim == 3:
                    self._pbasis[diff] = [
                        self._pbasis_create(i=i, j=j, k=k, dx=dx, dy=dy, dz=dz)
                        for i in range(maxdeg + 1)
                        for j in range(maxdeg + 1)
                        for k in range(maxdeg + 1)
                        if is_tensorial or i + j + k <= maxdeg
                    ]

    def _eval_dofs(self, mesh, tind=None):
        if tind is None:
            tind = np.arange(mesh.t.shape[1])

        N = len(self._pbasis[()])
        V = np.zeros((len(tind), N, N))
        w = {
            'v': np.array([mesh.p[:, mesh.t[itr, tind]]
                           for itr in range(mesh.t.shape[0])]),
        }
        if mesh.p.shape[0] >= 2:
            w['e'] = np.array([
                .5 * (w['v'][itr] + w['v'][(itr + 1) % mesh.t.shape[0]])
                for itr in range(mesh.t.shape[0])
            ])
            w['n'] = np.array([
                w['v'][itr] - w['v'][(itr + 1) % mesh.t.shape[0]]
                for itr in range(mesh.t.shape[0])
            ])
            w['n'][2] = -w['n'][2]  # direction swapped due to mesh numbering
            for itr in range(3):
                w['n'][itr] = np.array([w['n'][itr, 1, :],
                                        -w['n'][itr, 0, :]])
                w['n'][itr] /= np.linalg.norm(w['n'][itr], axis=0)

        # evaluate dofs, gdof implemented in subclasses
        for itr in range(N):
            for jtr in range(N):
                F = {k: self._pbasis[k][itr] for k in self._pbasis}
                V[:, jtr, itr] = self.gdof(F, w, jtr)

        return V
