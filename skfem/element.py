"""Element classes define and evaluate the finite element basis
functions.

:class:`~skfem.element.Element` objects are supplemented to the
constructors of :class:`~skfem.assembly.InteriorBasis` and
:class:`~skfem.assembly.FacetBasis` by the user.

>>> from skfem import *
>>> m = MeshTri()
>>> e = ElementTriP2()
>>> basis = InteriorBasis(m, e)

"""

import numpy as np

from typing import Optional, Tuple, Union, List

from numpy import ndarray

class Element():
    nodal_dofs: int = 0 
    facet_dofs: int = 0
    interior_dofs: int = 0
    edge_dofs: int = 0
    dim: int = -1
    maxdeg: int = -1
    # 0 - scalar, 1 - vector, 2 - tensor, etc
    order: Union[Tuple[int, int], Tuple[int, int, int]] = (-1, -1)
    dofnames: List[str] = []

    def orient(self, mapping, i, tind=None):
        """Orient basis functions. By default all = 1."""
        if tind is None:
            return 1 + 0*mapping.mesh.t[0, :]
        else:
            return 1 + 0*tind

    def gbasis(self,
               mapping,
               X: ndarray,
               i: int,
               tind: Optional[ndarray] = None) -> Union[Tuple[ndarray, ndarray],
                                                        Tuple[ndarray, ndarray, ndarray]]:
        """Evaluate the global basis functions, given local points X.

        The global points - at which the global basis is evaluated at -
        are defined through x = F(X), where F corresponds to the given mapping.

        Parameters
        ----------
        mapping
            Local-to-global mapping, an object of type :class:`~skfem.mapping.Mapping`.
        X
            An array of local points. The following shapes are supported: (Ndim x Npoints)
            and (Ndim x Nelems x Npoints), i.e. local points shared by all elements
            or different local points in each element.
        i
            Only the i'th basis function is evaluated.
        tind
            Optionally, choose a subset of elements to evaluate the basis at.

        Returns
        -------
        (u, du) or (u, du, ddu)
            The number of return arguments depends on the length of self.order.
            The shape of k'th return argument depends on the value of self.order[k].

        """
        raise NotImplementedError("Element must implement gbasis.")


class ElementH1(Element):
    order = (0, 1)

    def gbasis(self, mapping, X, i, tind=None):
        phi, dphi = self.lbasis(X, i)
        invDF = mapping.invDF(X, tind)
        if len(X.shape) == 2:
            return np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])),\
                   np.einsum('ijkl,il->jkl', invDF, dphi)
        elif len(X.shape) == 3:
            return np.broadcast_to(phi, (invDF.shape[2], invDF.shape[3])), \
                   np.einsum('ijkl,ikl->jkl', invDF, dphi)

    def lbasis(self, X, i):
        raise Exception("ElementH1 lbasis method not found.")


class ElementVectorH1(Element):
    order = (1, 2)

    def __init__(self, elem):
        self.dim = elem.dim
        self.elem = elem

        self.nodal_dofs = self.elem.nodal_dofs * self.dim
        self.facet_dofs = self.elem.facet_dofs * self.dim
        self.interior_dofs = self.elem.interior_dofs * self.dim
        self.edge_dofs = self.elem.edge_dofs * self.dim

        self.dofnames = [i + "^" + str(j+1) for i in elem.dofnames for j in range(self.dim)]
        self.maxdeg = elem.maxdeg

    def gbasis(self, mapping, X, i, tind=None):
        ind = int(np.floor(float(i)/float(self.dim)))
        n = i - self.dim*ind
        phi, dphi = self.elem.gbasis(mapping, X, ind, tind)
        u = np.zeros((self.dim,) + phi.shape)
        du = np.zeros((self.dim,) + dphi.shape)
        u[n] = phi
        du[n] = dphi
        return u, du


class ElementHdiv(Element):
    order = (1, 0)

    def orient(self, mapping, i, tind=None):
        if tind is not None:
            # TODO fix
            raise NotImplementedError("TODO: fix tind support in ElementHdiv")
        return -1 + 2*(mapping.mesh.f2t[0, mapping.mesh.t2f[i, :]] \
                       == np.arange(mapping.mesh.t.shape[1]))

    def gbasis(self, mapping, X, i, tind=None):
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        return np.einsum('ijkl,jl,kl->ikl', DF, phi, 1/np.abs(detDF)*orient[:, None]),\
               dphi/(np.abs(detDF)*orient[:, None])

    def lbasis(self, X, i):
        raise Exception("ElementHdiv lbasis method not found.")


class ElementHcurl(Element):
    """Note: only 3D support. Piola transformation
    is different in 2D."""
    
    order = (1, 1)

    def orient(self, mapping, i, tind=None):
        if tind is not None:
            # TODO fix
            raise NotImplementedError("TODO: fix tind support in ElementHcurl")
        t1 = [0, 1, 0, 0, 1, 2][i]
        t2 = [1, 2, 2, 3, 3, 3][i]
        return 1 - 2*(mapping.mesh.t[t1, :] > mapping.mesh.t[t2, :])

    def gbasis(self, mapping, X, i, tind=None):
        phi, dphi = self.lbasis(X, i)
        DF = mapping.DF(X, tind)
        invDF = mapping.invDF(X, tind)
        detDF = mapping.detDF(X, tind)
        orient = self.orient(mapping, i, tind)
        return np.einsum('ijkl,il,k->jkl', invDF, phi, orient), \
               np.einsum('ijkl,jl,kl->ikl', DF, dphi, 1/detDF*orient[:, None])

    def lbasis(self, X, i):
        raise Exception("ElementHcurl lbasis method not found.")


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
        return eval("lambda x, y: {}*x**{}*y**{}".format(cx*cy, np.max([i-dx, 0]), np.max([j-dy, 0])))

    def _pbasis_init(self, N):
        """Define power bases (for 2D)."""
        if not hasattr(self, '_pbasis'):
            setattr(self, '_pbasis', [self._pbasis_create_xy(i, j) for i in range(N+1) for j in range(N+1) if i + j <= N])
            setattr(self, '_pbasisdx', [self._pbasis_create_xy(i, j, dx=1) for i in range(N+1) for j in range(N+1) if i + j <= N])
            setattr(self, '_pbasisdy', [self._pbasis_create_xy(i, j, dy=1) for i in range(N+1) for j in range(N+1) if i + j <= N])
            setattr(self, '_pbasisdxx', [self._pbasis_create_xy(i, j, dx=2) for i in range(N+1) for j in range(N+1) if i + j <= N])
            setattr(self, '_pbasisdxy', [self._pbasis_create_xy(i, j, dx=1, dy=1) for i in range(N+1) for j in range(N+1) if i + j <= N])
            setattr(self, '_pbasisdyy', [self._pbasis_create_xy(i, j, dy=2) for i in range(N+1) for j in range(N+1) if i + j <= N])

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
            e[0] = 0.5*(v[0] + v[1])
            e[1] = 0.5*(v[1] + v[2])
            e[2] = 0.5*(v[0] + v[2])

            # normal vectors
            n[0] = v[0] - v[1]
            n[1] = v[1] - v[2]
            n[2] = v[0] - v[2]

            for itr in range(3):
                n[itr] = np.array([n[itr, 1, :], -n[itr, 0, :]])
                n[itr] /= np.linalg.norm(n[itr], axis=0)
        else:
            raise NotImplementedError("The used mesh type not supported in ElementH2.")

        # evaluate dofs, gdof implemented in subclasses
        for itr in range(N):
            for jtr in range(N):
                u = self._pbasis[itr]
                du = [self._pbasisdx[itr], self._pbasisdy[itr]]
                ddu = [self._pbasisdxx[itr], self._pbasisdxy[itr], self._pbasisdyy[itr]]
                V[:, jtr, itr] = self.gdof(u, du, ddu, v, e, n, jtr)

        return V

# Triangular

class ElementTriP1(ElementH1):
    nodal_dofs = 1
    dim = 2
    maxdeg = 1
    dofnames = ['u']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = 1 - x - y
            dphi = np.array([-1 + 0*x, -1 + 0*x])
        elif i == 1:
            phi = x
            dphi = np.array([1 + 0*x, 0*x])
        elif i == 2:
            phi = y
            dphi = np.array([0*x, 1 + 0*x])
        else:
            raise Exception("!")

        return phi, dphi


class ElementTriP2(ElementH1):
    nodal_dofs = 1
    facet_dofs = 1
    dim = 2
    maxdeg = 2
    dofnames = ['u', 'u']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = 1-3*x-3*y+2*x**2+4*x*y+2*y**2
            dphi = np.array([-3+4*x+4*y, -3+4*x+4*y])
        elif i == 1:
            phi = 2*x**2-x
            dphi = np.array([4*x-1, 0*x])
        elif i == 2:
            phi = 2*y**2-y
            dphi = np.array([0*x, 4*y-1])
        elif i == 3: # 0->1
            phi = 4*x-4*x**2-4*x*y
            dphi = np.array([4-8*x-4*y, -4*x])
        elif i == 4: # 1->2
            phi = 4*x*y
            dphi = np.array([4*y, 4*x])
        elif i == 5: # 0->2
            phi = 4*y-4*x*y-4*y**2
            dphi = np.array([-4*y, 4-4*x-8*y])
        else:
            raise Exception("!")

        return phi, dphi


class ElementTriDG(ElementH1):
    dim = 2

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = 3*elem.nodal_dofs + 3*elem.facet_dofs + elem.interior_dofs

    def lbasis(self, X, i):
        return self.elem.lbasis(X, i)


class ElementTriP0(ElementH1):
    interior_dofs = 1
    dim = 2
    maxdeg = 0
    dofnames = ['u']

    def lbasis(self, X, i):
        return 1 + 0*X[0, :], 0*X


class ElementTriRT0(ElementHdiv):
    facet_dofs = 1
    dim = 2
    maxdeg = 1
    dofnames = ['u_n']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = np.array([x, y-1])
            dphi = 2 + 0*x
        elif i == 1:
            phi = np.array([x, y])
            dphi = 2 + 0*x
        elif i == 2:
            phi = np.array([x-1, y])
            dphi = 2 + 0*x
        else:
            raise Exception("!")

        return phi, dphi


class ElementTriMorley(ElementH2):
    nodal_dofs = 1
    facet_dofs = 1
    dim = 2
    maxdeg = 2
    dofnames = ['u', 'u_n']

    def gdof(self, u, du, ddu, v, e, n, i):
        if i == 0:
            return u(*v[0])
        elif i == 1:
            return u(*v[1])
        elif i == 2:
            return u(*v[2])
        elif i == 3:
            return du[0](*e[0])*n[0, 0] + du[1](*e[0])*n[0, 1]
        elif i == 4:
            return du[0](*e[1])*n[1, 0] + du[1](*e[1])*n[1, 1]
        elif i == 5:
            return du[0](*e[2])*n[2, 0] + du[1](*e[2])*n[2, 1]
        else:
            raise Exception("!")


class ElementTriArgyris(ElementH2):
    nodal_dofs = 6
    facet_dofs = 1
    dim = 2
    maxdeg = 5
    dofnames = ['u', 'u_x', 'u_y', 'u_xx', 'u_xy', 'u_yy', 'u_n']

    def gdof(self, u, du, ddu, v, e, n, i):
        if i < 18:
            j = i % 6
            k = int(i/6)
            if j == 0:
                return u(*v[k])
            elif j == 1:
                return du[0](*v[k])
            elif j == 2:
                return du[1](*v[k])
            elif j == 3:
                return ddu[0](*v[k])
            elif j == 4:
                return ddu[1](*v[k])
            elif j == 5:
                return ddu[2](*v[k])
        elif i == 18:
            return du[0](*e[0])*n[0, 0] + du[1](*e[0])*n[0, 1]
        elif i == 19:
            return du[0](*e[1])*n[1, 0] + du[1](*e[1])*n[1, 1]
        elif i == 20:
            return du[0](*e[2])*n[2, 0] + du[1](*e[2])*n[2, 1]

# Quadilateral

class ElementQuad1(ElementH1):
    nodal_dofs = 1
    dim = 2
    maxdeg = 2
    dofnames = ['u']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = 0.25*(1 - x)*(1 - y)
            dphi = np.array([0.25*(-1 + y), 0.25*(-1 + x)])
        elif i == 1:
            phi = 0.25*(1 + x)*(1 - y)
            dphi = np.array([0.25*(1 - y), 0.25*(-1 - x)])
        elif i == 2:
            phi = 0.25*(1 + x)*(1 + y)
            dphi = np.array([0.25*(1 + y), 0.25*(1 + x)])
        elif i == 3:
            phi = 0.25*(1 - x)*(1 + y)
            dphi = np.array([0.25*(-1 - y), 0.25*(1 - x)])
        else:
            raise Exception("!")

        return phi, dphi

class ElementQuad2(ElementH1):
    nodal_dofs = 1
    facet_dofs = 1
    interior_dofs = 1
    dim = 2
    maxdeg = 3
    dofnames = ['u', 'u', 'u']

    def lbasis(self, X, i):
        x, y = X[0, :], X[1, :]

        if i == 0:
            phi = 0.25*(x**2-x)*(y**2-y)
            dphi = np.array([((-1 + 2*x)*(-1 + y)*y)/4., ((-1 + x)*x*(-1 + 2*y))/4.])
        elif i == 1:
            phi = 0.25*(x**2+x)*(y**2-y)
            dphi = np.array([((1 + 2*x)*(-1 + y)*y)/4.,(x*(1 + x)*(-1 + 2*y))/4. ])
        elif i == 2:
            phi = 0.25*(x**2+x)*(y**2+y)
            dphi = np.array([((1 + 2*x)*y*(1 + y))/4., (x*(1 + x)*(1 + 2*y))/4.])
        elif i == 3:
            phi = 0.25*(x**2-x)*(y**2+y)
            dphi = np.array([((-1 + 2*x)*y*(1 + y))/4., ((-1 + x)*x*(1 + 2*y))/4.])
        elif i == 4:
            phi = 0.5*(y**2-y)*(1-x**2)
            dphi = np.array([-(x*(-1 + y)*y), -((-1 + x**2)*(-1 + 2*y))/2.])
        elif i == 5:
            phi = 0.5*(x**2+x)*(1-y**2)
            dphi = np.array([-((1 + 2*x)*(-1 + y**2))/2., -(x*(1 + x)*y)])
        elif i == 6:
            phi = 0.5*(y**2+y)*(1-x**2)
            dphi = np.array([-(x*y*(1 + y)), -((-1 + x**2)*(1 + 2*y))/2.])
        elif i == 7:
            phi = 0.5*(x**2-x)*(1-y**2)
            dphi = np.array([-((-1 + 2*x)*(-1 + y**2))/2., -((-1 + x)*x*y)])
        elif i == 8:
            phi = (1-x**2)*(1-y**2)
            dphi = np.array([2*x*(-1 + y**2), 2*(-1 + x**2)*y])
        else:
            raise Exception("!")

        return phi, dphi

# Tetrahedral

class ElementTetP0(ElementH1):
    interior_dofs = 1
    dim = 3
    maxdeg = 0
    dofnames = ['u']

    def lbasis(self, X, i):
        return 1 + 0*X[0, :], 0*X


class ElementTetP1(ElementH1):
    nodal_dofs = 1
    dim = 3
    maxdeg = 1
    dofnames = ['u']

    def lbasis(self, X, i):
        x, y, z = X[0, :], X[1, :], X[2, :]

        if i == 0:
            phi = 1 - x - y - z
            dphi = np.array([-1 + 0*x, -1 + 0*x, -1 + 0*x])
        elif i == 1:
            phi = x
            dphi = np.array([1 + 0*x, 0*x, 0*x])
        elif i == 2:
            phi = y
            dphi = np.array([0*x, 1 + 0*x, 0*x])
        elif i == 3:
            phi = z 
            dphi = np.array([0*x, 0*x, 1 + 0*x])
        else:
            raise Exception("!")

        return phi, dphi


class ElementTetP2(ElementH1):
    nodal_dofs = 1
    edge_dofs = 1
    dim = 3
    maxdeg = 2
    dofnames = ['u', 'u']

    def lbasis(self, X, i):
        x, y, z = X[0, :], X[1, :], X[2, :]

        if i == 0:
            phi = 1 - x - y - z
            dphi = np.array([-1 + 0*x, -1 + 0*x, -1 + 0*x])
        elif i == 1:
            phi = x
            dphi = np.array([1 + 0*x, 0*x, 0*x])
        elif i == 2:
            phi = y
            dphi = np.array([0*x, 1 + 0*x, 0*x])
        elif i == 3:
            phi = z 
            dphi = np.array([0*x, 0*x, 1 + 0*x])
        elif i == 4: # (0, 1)
            phi = (1 - x - y - z)*x
            dphi = np.array([1 - 2*x - y - z, -x, -x])
        elif i == 5: # (1, 2)
            phi = x*y
            dphi = np.array([y, x, 0*x])
        elif i == 6: # (0, 2)
            phi = (1 - x - y - z)*y
            dphi = np.array([-y, 1 - x - 2*y - z, -y])
        elif i == 7: # (0, 3)
            phi = (1 - x - y - z)*z
            dphi = np.array([-z, -z, 1 - x - y - 2*z])
        elif i == 8: # (1, 3)
            phi = x*z
            dphi = np.array([z, 0*x, x])
        elif i == 9: # (2, 3)
            phi = y*z
            dphi = np.array([0*x, z, y])
        else:
            raise Exception("!")

        return phi, dphi


class ElementTetRT0(ElementHdiv):
    facet_dofs = 1
    dim = 3
    maxdeg = 1
    dofnames = ['u_n']

    def lbasis(self, X, i):
        x, y, z = X[0, :], X[1, :], X[2, :]

        if i == 0:
            phi = np.array([x, y, z-1])
            dphi = 3 + 0*x
        elif i == 1:
            phi = np.array([x, y-1, z])
            dphi = 3 + 0*x
        elif i == 2:
            phi = np.array([x-1, y, z])
            dphi = 3 + 0*x
        elif i == 3:
            phi = np.array([x, y, z])
            dphi = 3 + 0*x
        else:
            raise Exception("!")

        return phi, dphi


class ElementTetN0(ElementHcurl):
    edge_dofs = 1
    dim = 3
    maxdeg = 1
    dofnames = ['u_t']

    def lbasis(self, X, i):
        x, y, z = X[0, :], X[1, :], X[2, :]

        if i == 0:
            phi = np.array([1-z-y, x, x])
            dphi = np.array([0*x, -2 + 0*x, 2 + 0*x])
        elif i == 1:
            phi = np.array([-y, x, 0*z])
            dphi = np.array([0*x, 0*x, 2 + 0*x])
        elif i == 2:
            phi = np.array([y, 1-z-x, y])
            dphi = np.array([2 + 0*x, 0*x, -2 + 0*x])
        elif i == 3:
            phi = np.array([z, z, 1-x-y])
            dphi = np.array([-2 + 0*x, 2 + 0*x, 0*x])
        elif i == 4:
            phi = np.array([-z, 0*y, x])
            dphi = np.array([0*x, -2 + 0*x, 0*x])
        elif i == 5:
            phi = np.array([0*x, -z, y])
            dphi = np.array([2 + 0*x, 0*x, 0*x])
        else:
            raise Exception("!")

        return phi, dphi

# Hexahedral

class ElementHex1(ElementH1):
    nodal_dofs = 1
    dim = 3
    maxdeg = 3
    dofnames = ['u']

    def lbasis(self, X, i):
        x, y, z = X[0, :], X[1, :], X[2, :]

        if i == 0:
            phi = 0.125*(1 - x)*(1 - y)*(1 - z)
            dphi = np.array([0.125*(-1 + y)*(1 - z),
                             0.125*(-1 + x)*(1 - z),
                             -0.125*(1 - x)*(1 - y)])
        elif i == 1:
            phi = 0.125*(1 + x)*(1 - y)*(1 - z)
            dphi = np.array([0.125*(1 - y)*(1 - z),
                             0.125*(-1 - x)*(1 - z),
                             -0.125*(1 + x)*(1 - y)])
        elif i == 2:
            phi = 0.125*(1 + x)*(1 + y)*(1 - z)
            dphi = np.array([0.125*(1 + y)*(1 - z),
                             0.125*(1 + x)*(1 - z),
                             -0.125*(1 + x)*(1 + y)])
        elif i == 3:
            phi = 0.125*(1 - x)*(1 + y)*(1 - z)
            dphi = np.array([0.125*(-1 - y)*(1 - z),
                             0.125*(1 - x)*(1 - z),
                             -0.125*(1 - x)*(1 + y)])
        elif i == 4:
            phi = 0.125*(1 - x)*(1 - y)*(1 + z)
            dphi = np.array([0.125*(-1 + y)*(1 + z),
                             0.125*(-1 + x)*(1 + z),
                             0.125*(1 - x)*(1 - y)])
        elif i == 5:
            phi = 0.125*(1 + x)*(1 - y)*(1 + z)
            dphi = np.array([0.125*(1 - y)*(1 + z),
                             0.125*(-1 - x)*(1 + z),
                             0.125*(1 + x)*(1 - y)])
        elif i == 6:
            phi = 0.125*(1 + x)*(1 + y)*(1 + z)
            dphi = np.array([0.125*(1 + y)*(1 + z),
                             0.125*(1 + x)*(1 + z),
                             0.125*(1 + x)*(1 + y)])
        elif i == 7:
            phi = 0.125*(1 - x)*(1 + y)*(1 + z)
            dphi = np.array([0.125*(-1 - y)*(1 + z),
                             0.125*(1 - x)*(1 + z),
                             0.125*(1 - x)*(1 + y)])
        else:
            raise Exception("!")

        return phi, dphi

# 1D elements

class ElementLineP1(ElementH1):
    nodal_dofs = 1
    dim = 1
    maxdeg = 1
    dofnames = ['u']

    def lbasis(self, X, i):
        x = X[0, :]

        if i == 0:
            phi = 1 - x
            dphi = np.array([-1 + 0*x])
        elif i == 1:
            phi = x
            dphi = np.array([1 + 0*x])
        else:
            raise Exception("!")

        return phi, dphi

