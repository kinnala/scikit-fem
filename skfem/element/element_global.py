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

        if (self.V is None
                or getattr(self, '_V_mesh', None) is not mapping.mesh):
            # initialize power basis
            self._pbasis_init(self.maxdeg,
                              self.dim,
                              self.derivatives,
                              self.tensorial_basis)
            # construct Vandermonde matrix and invert it
            self.V = np.linalg.inv(self._eval_dofs(mapping.mesh))
            self._V_mesh = mapping.mesh

        # the basis is evaluated for all i at once and cached for the
        # subsequent calls with the same arguments
        cache = getattr(self, '_gbasis_cache', None)
        if (cache is None
                or cache[0] is not mapping.mesh
                or not np.array_equal(cache[1], X)
                or not np.array_equal(cache[2], tind)):
            cache = (mapping.mesh, np.copy(X), np.copy(tind),
                     self._gbasis_all(mapping, X, tind))
            self._gbasis_cache = cache
        U = [u[i] for u in cache[3]]
        if i == len(cache[3][0]) - 1:
            # release the memory after the last basis function
            self._gbasis_cache = None

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

    def _gbasis_all(self, mapping, X, tind):
        """Evaluate all basis functions and their derivatives."""
        x = mapping.F(X, tind=tind)
        h = self._h[tind]
        y = (x - self._c[:, tind][:, :, None]) / h[:, None]
        # (element, basis function, power basis function)
        V = self.V[tind].transpose(0, 2, 1)
        nelems, N, _ = V.shape
        nqp = x.shape[-1]

        U = []
        P = np.empty((nelems, N, nqp))
        for k in range(self.derivatives + 1):
            diffs = list(itertools.product(*((list(range(self.dim)),) * k)))
            Vk = V / h[:, None, None] ** k
            Uk = np.empty((N, len(diffs), nelems, nqp))
            for d, diff in enumerate(diffs):
                # evaluate the power basis only once for all basis functions
                for j, p in enumerate(self._pbasis[diff]):
                    P[:, j] = p(*y)
                Uk[:, d] = np.matmul(Vk, P).transpose(1, 0, 2)
            U.append(Uk.reshape((N,) + (self.dim,) * k + (nelems, nqp)))

        return U

    def _local(self, p, order):
        """Evaluate a power basis function given in local coordinates."""
        c, h = self._c, self._h

        def fun(*x):
            y = ((x[j] - c[j]) / h for j in range(self.dim))
            return p(*y) / h ** order

        return fun

    @staticmethod
    def _monomial(c, *exps):
        """Return a monomial as a function.

        The powers are written as products: numpy evaluates, e.g., x ** 3
        an order of magnitude slower if x is negative.

        """
        args = ['x', 'y', 'z'][:len(exps)]
        factors = [a for a, n in zip(args, exps) for _ in range(max(n, 0))]
        if len(factors) == 0:
            # return an array of the correct shape
            return eval("lambda {}: {} + 0. * x".format(', '.join(args), c))
        return eval("lambda {}: {}*{}".format(', '.join(args),
                                              c,
                                              '*'.join(factors)))

    def _pbasis_create(self, i, j=None, k=None, dx=0, dy=0, dz=0):
        """Return a single power basis function."""
        if j is None and k is None:  # 1d
            cx = 1
            if dx > 0:
                for l in np.arange(dx, 0, -1):
                    cx *= i - dx + l
            return self._monomial(cx, i - dx)
        elif k is None:  # 2d
            cx = 1
            cy = 1
            if dx > 0:
                for l in np.arange(dx, 0, -1):
                    cx *= i - dx + l
            if dy > 0:
                for l in np.arange(dy, 0, -1):
                    cy *= j - dy + l
            return self._monomial(cx * cy, i - dx, j - dy)
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
            return self._monomial(cx * cy * cz, i - dx, j - dy, k - dz)

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
                dy = sum([1 for d in diff if d == 1]) if dim >= 2 else None
                dz = sum([1 for d in diff if d == 2]) if dim >= 3 else None
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
        if self.dim in (1, 2):
            # Periodic topology need not index the physical geometry nodes.
            vertices = mesh.mapping().F(self.refdom.p, tind=tind)
            vertices = vertices.transpose(2, 0, 1)
        else:
            vertices = np.array([mesh.p[:, mesh.t[itr, tind]]
                                 for itr in range(mesh.t.shape[0])])
        # the power basis is defined in local coordinates (x - c) / h for
        # better conditioning of the Vandermonde matrix
        self._c = np.mean(vertices, axis=0)
        self._h = np.max(np.linalg.norm(vertices - self._c, axis=1), axis=0)
        w = {'v': vertices, 'tix': mesh.t[:, tind]}
        if mesh.p.shape[0] == 2:
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

            if mesh.t.shape[0] == 3:
                # Identified periodic vertices can reverse a local edge's
                # direction. Use the first adjacent cell's endpoint ordering
                # on both sides so that a shared normal DOF has one direction.
                facets = mesh.t2f[:, tind]
                cells = mesh.f2t[0, facets]
                local = np.argmax(mesh.t2f[:, cells] == facets[None], axis=0)
                edges = np.array(self.refdom.facets)
                first = (mesh.t[edges[local, 0], cells]
                         < mesh.t[edges[local, 1], cells])
                current = (mesh.t[edges[:, 0, None], tind]
                           < mesh.t[edges[:, 1, None], tind])
                w['n'] *= np.where(first == current, 1., -1.)[:, None, :]

            # Evaluate boundary normals directly in the adjacent cell.
            # Facet geometry may use different node indices on a DG mesh.
            ix = np.isin(mesh.t2f[:, tind], mesh.boundary_facets())
            if np.any(ix):
                from skfem.quadrature import get_quadrature

                sortix = np.argsort(mesh.t2f[:, tind][ix])
                find = mesh.t2f[:, tind][ix][sortix]
                cells = mesh.f2t[0, find]
                local = np.array([
                    np.argmax(mesh.t[:, cells] == nodes, axis=0)
                    for nodes in mesh.facets[:, find]
                ])
                # Preserve the first facet quadrature point and endpoint
                # order, including the normal projection on curved cells.
                s = get_quadrature(mesh.brefdom, 0)[0][0, 0]
                X = ((1. - s) * self.refdom.p[:, local[0]]
                     + s * self.refdom.p[:, local[1]])[:, :, None]
                normals = mesh.mapping().normals(
                    X, cells, find, mesh.t2f)[:, :, 0]
                original = np.array([w['n'][:, i, :][ix][sortix]
                                     for i in range(2)])
                signs = np.sum(original * normals, axis=0)
                signs1 = signs.copy()
                signs1[sortix] = signs
                w['n'][:, 0, :][ix] *= signs1
                w['n'][:, 1, :][ix] *= signs1

        # evaluate dofs, gdof implemented in subclasses
        for itr in range(N):
            for jtr in range(N):
                F = {k: self._local(self._pbasis[k][itr], len(k))
                     for k in self._pbasis}
                V[:, jtr, itr] = self.gdof(F, w, jtr)

        return V
