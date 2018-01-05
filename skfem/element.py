# -*- coding: utf-8 -*-
"""
The finite element definitions.

Try for example the following actual implementations:
    * :class:`skfem.element.ElementLocalTriP1`
    * :class:`skfem.element.ElementLocalTriP2`
"""
import numpy as np
from skfem.utils import const_cell

class ElementLocal(object):
    """A finite element defined through basis functions."""

    maxdeg = 0 #: Maximum polynomial degree
    dim = 0 #: Spatial dimension

    n_dofs = 0 #: Number of nodal dofs
    i_dofs = 0 #: Number of interior dofs
    f_dofs = 0 #: Number of facet dofs (2d and 3d only)
    e_dofs = 0 #: Number of edge dofs (3d only)

    def lbasis(self, X, i):
        """Returns local basis functions evaluated at some local points."""
        raise NotImplementedError("Local basis (lbasis) not implemented!")

    def gbasis(self, X, i, tind):
        """Returns global basis functions evaluated at some local points."""
        raise NotImplementedError("Global basis (gbasis) not implemented!")

class ElementGlobal(object):
    """A finite element defined through DOF functionals."""

    maxdeg = 0 #: Maximum polynomial degree
    dim = 0 #: Spatial dimension

    n_dofs = 0 #: Number of nodal dofs
    i_dofs = 0 #: Number of interior dofs
    f_dofs = 0 #: Number of facet dofs (2d and 3d only)
    e_dofs = 0 #: Number of edge dofs (3d only)

    def _evaldofs(self, mesh, tind=None):
        if tind is None:
            tind = np.arange(mesh.t.shape[1])
        N=len(self._pbasis)

        V=np.zeros((len(tind), N, N))

        if mesh.t.shape[0]==2:
            v1 = mesh.p[:, mesh.t[0, tind]][0, :]
            v2 = mesh.p[:, mesh.t[1, tind]][0, :]
            dofvars = {
                'v1': v1,
                'v2': v2,
            }
        elif mesh.t.shape[0]==3:
            v1 = mesh.p[:, mesh.t[0, tind]]
            v2 = mesh.p[:, mesh.t[1, tind]]
            v3 = mesh.p[:, mesh.t[2, tind]]

            e1 = 0.5*(v1 + v2)
            e2 = 0.5*(v2 + v3)
            e3 = 0.5*(v1 + v3)

            t1 = v1 - v2
            t2 = v2 - v3
            t3 = v1 - v3

            n1 = np.array([t1[1, :], -t1[0, :]])
            n2 = np.array([t2[1, :], -t2[0, :]])
            n3 = np.array([t3[1, :], -t3[0, :]])

            n1 /= np.linalg.norm(n1, axis=0)
            n2 /= np.linalg.norm(n2, axis=0)
            n3 /= np.linalg.norm(n3, axis=0)

            dofvars={
                'v1': v1,
                'v2': v2,
                'v3': v3,
                'e1': e1,
                'e2': e2,
                'e3': e3,
                'n1': n1,
                'n2': n2,
                'n3': n3,
                }
        else:
            raise NotImplementedError("The used mesh type not supported in ElementGlobal.")

        # evaluate dofs
        for itr in range(N):
            for jtr in range(N):
                V[:, jtr, itr] = self.gdof(dofvars, itr, jtr)

        return V

    def evalbasis(self, mesh, qps, tind=None):
        # initialize power basis
        self._pbasisNinit(self.dim, self.maxdeg)
        N = len(self._pbasis)

        # construct Vandermonde matrix and invert it
        V = self._evaldofs(mesh, tind=tind)
        V = np.linalg.inv(V)

        # initialize
        if self.dim==1:
            u = const_cell(0 * qps, N)
            du = const_cell(0*qps, N)
            ddu = const_cell(0*qps, N)
            d4u = const_cell(0*qps, N)
        elif self.dim==2:
            u = const_cell(0 * qps[0], N)
            du = const_cell(0*qps[0], N, self.dim)
            ddu = const_cell(0*qps[0], N, self.dim, self.dim)
            d4u = const_cell(0*qps[0], N, self.dim, self.dim)

        # loop over new basis
        for jtr in range(N):
            # loop over power basis
            for itr in range(N):
                if self.dim==1:
                    u[jtr] += V[:, itr, jtr][:, None] \
                              * self._pbasis[itr](qps)
                    du[jtr] += V[:, itr, jtr][:, None] \
                                  * self._pbasisdx[itr](qps)
                    ddu[jtr] += V[:, itr, jtr][:, None] \
                                      * self._pbasisdxx[itr](qps)
                    d4u[jtr] += V[:, itr, jtr][:, None] \
                                      * self._pbasisdxxxx[itr](qps)
                elif self.dim==2:
                    u[jtr] += V[:, itr, jtr][:, None]\
                              * self._pbasis[itr](qps[0], qps[1])
                    du[jtr][0] += V[:, itr, jtr][:, None]\
                                  * self._pbasisdx[itr](qps[0], qps[1])
                    du[jtr][1] += V[:, itr, jtr][:,None]\
                                  * self._pbasisdy[itr](qps[0], qps[1])
                    ddu[jtr][0][0] += V[:, itr, jtr][:, None]\
                                      * self._pbasisdxx[itr](qps[0], qps[1])
                    ddu[jtr][0][1] += V[:, itr, jtr][:, None]\
                                      * self._pbasisdxy[itr](qps[0], qps[1])
                    ddu[jtr][1][1] += V[:, itr, jtr][:, None]\
                                      * self._pbasisdyy[itr](qps[0], qps[1])
                    d4u[jtr][0][0] += V[:, itr, jtr][:, None]\
                                      * self._pbasisdxxxx[itr](qps[0], qps[1])
                    d4u[jtr][1][1] += V[:, itr, jtr][:, None]\
                                      * self._pbasisdyyyy[itr](qps[0], qps[1])
                    d4u[jtr][0][1] += V[:, itr, jtr][:, None]\
                                      * self._pbasisdxxyy[itr](qps[0], qps[1])
                else:
                    raise NotImplementedError("!")
            if self.dim==2:
                ddu[jtr][1][0] = ddu[jtr][0][1]
                d4u[jtr][1][0] = d4u[jtr][0][1]
        return u, du, ddu, d4u

    def _pbasisNinit(self, dim, N):
        """Define power bases."""
        if not hasattr(self, '_pbasis' + str(N)):
            import sympy as sp
            from sympy.abc import x, y, z
            R = list(range(N+1))
            if dim==1:
                ops = {
                    '': lambda a: a,
                    'dx': lambda a: sp.diff(a, x),
                    'dxx': lambda a: sp.diff(a, x, 2),
                    'dxxxx': lambda a: sp.diff(a, x, 4),
                }
                for name, op in ops.items():
                    pbasis = [sp.lambdify((x), op(x**i), "numpy")
                              for i in R]
                    # workaround for constant shape bug in SymPy
                    for itr in range(len(pbasis)):
                        const = pbasis[itr](np.zeros(1))
                        if type(const) is int:
                            pbasis[itr] = lambda X, const=const: const*np.ones(X.shape)
                    setattr(self,'_pbasis'+name, pbasis)
            elif dim==2:
                ops = {
                    '': lambda a: a,
                    'dx': lambda a: sp.diff(a, x),
                    'dy': lambda a: sp.diff(a, y),
                    'dxx': lambda a: sp.diff(a, x, 2),
                    'dyy': lambda a: sp.diff(a, y, 2),
                    'dxy': lambda a: sp.diff(sp.diff(a, x), y),
                    'dxxxx': lambda a: sp.diff(a, x, 4),
                    'dyyyy': lambda a: sp.diff(a, y, 4),
                    'dxxyy': lambda a: sp.diff(sp.diff(a, x, 2), y, 2),
                }
                for name, op in ops.items():
                    pbasis = [sp.lambdify((x,y), op(x**i*y**j), "numpy")
                              for i in R for j in R if i+j<=N]
                    # workaround for constant shape bug in SymPy
                    for itr in range(len(pbasis)):
                        const = pbasis[itr](np.zeros(2), np.zeros(2))
                        if type(const) is int:
                            pbasis[itr] = lambda X, Y, const=const: const*np.ones(X.shape)
                    setattr(self,'_pbasis'+name, pbasis)
            else:
                raise NotImplementedError("The given dimension not implemented!")

class ElementGlobalTriDG(ElementGlobal):
    """Transform all dofs into interior dofs."""

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.i_dofs = 3*elem.n_dofs + 3*elem.f_dofs + elem.i_dofs
        self.dim = 2
        elem._pbasisNinit(self.dim, self.maxdeg)

    def gdof(self, v, i, j):
        return self.elem.gdof(v, i, j)

class ElementGlobalTriP0(ElementGlobal):
    """Piecewise constant triangles."""

    dim = 2

    def __init__(self):
        self.maxdeg = 0
        self.n_dofs = 0
        self.f_dofs = 0
        self.i_dofs = 1
    def gdof(self, v, i, j):
        return [
                lambda: self._pbasis[i]((v['v1'][0, :] + v['v2'][0, :] + v['v3'][0, :])/3.0, (v['v1'][1, :] + v['v2'][1, :] + v['v3'][1, :])/3.0),
                ][j]()

class ElementGlobalLineP1(ElementGlobal):
    """C^1 continuous Hermite element for Euler-Bernoulli beam."""

    dim = 1

    def __init__(self):
        self.maxdeg = 1
        self.n_dofs = 1

    def gdof(self, v, i, j):
        return [
                lambda: self._pbasis[i](v['v1']),
                lambda: self._pbasis[i](v['v2']),
                ][j]()

class ElementGlobalLineHermite(ElementGlobal):
    """C^1 continuous Hermite element for Euler-Bernoulli beam."""

    dim = 1

    def __init__(self):
        self.maxdeg = 3
        self.n_dofs = 2

    def gdof(self, v, i, j):
        return [
                lambda: self._pbasis[i](v['v1']),
                lambda: self._pbasisdx[i](v['v1']),
                lambda: self._pbasis[i](v['v2']),
                lambda: self._pbasisdx[i](v['v2']),
                ][j]()

class ElementGlobalTriPp(ElementGlobal):
    """Triangular Pp element, Lagrange DOFs."""

    dim = 2

    def __init__(self, p=1):
        if p<1:
            raise NotImplementedError("Degree p<1 not supported.")

        self.p = p
        self.maxdeg = p

        self.n_dofs = 1
        self.f_dofs = np.max([p - 1, 0])
        self.i_dofs = np.max([(p - 1)*(p - 2)/2, 0])

        self.nbdofs = 3*self.n_dofs + 3*self.f_dofs + self.i_dofs

    def gdof(self, v, i, j):
        if j < 3: # vertex dofs
            return [
                    lambda: self._pbasis[i](v['v1'][0, :], v['v1'][1, :]),
                    lambda: self._pbasis[i](v['v2'][0, :], v['v2'][1, :]),
                    lambda: self._pbasis[i](v['v3'][0, :], v['v3'][1, :]),
                    ][j]()
        elif j < 3*self.f_dofs + 3: # edge dofs
            j = j - 3
            # generate node locations on edge
            points = np.linspace(0, 1, self.p + 1)
            points = points[1:-1]
            if j < self.f_dofs: # edge 1->2
                return self._pbasis[i](points[j]*v['v1'][0, :]
                                       + (1 - points[j])*v['v2'][0, :],
                                       points[j]*v['v1'][1, :]
                                       + (1 - points[j])*v['v2'][1, :])
            elif j < 2*self.f_dofs: # edge 2->3
                j = j - self.f_dofs
                return self._pbasis[i](points[j]*v['v2'][0, :]
                                       + (1 - points[j])*v['v3'][0, :],
                                       points[j]*v['v2'][1, :]
                                       + (1 - points[j])*v['v3'][1, :])
            else: # edge 1->3
                j = j - 2*self.f_dofs
                return self._pbasis[i](points[j]*v['v1'][0, :]
                                       + (1 - points[j])*v['v3'][0, :],
                                       points[j]*v['v1'][1, :]
                                       + (1 - points[j])*v['v3'][1, :])
        else: # interior dofs
            if self.i_dofs>1:
                raise NotImplementedError("TODO fix i_dofs for p>3")
            j = j - 3 - 3*self.f_dofs

            return self._pbasis[i]((v['v1'][0, :]+v['v2'][0, :]+v['v3'][0, :])/3,
                                   (v['v1'][1, :]+v['v2'][1, :]+v['v3'][1, :])/3)


class ElementGlobalMorley(ElementGlobal):
    """Morley element for fourth-order problems."""

    n_dofs = 1
    f_dofs = 1
    dim = 2
    maxdeg = 2

    def gdof(self, v, i, j):
        return [
            lambda: self._pbasis[i](v['v1'][0, :], v['v1'][1, :]),
            lambda: self._pbasis[i](v['v2'][0, :], v['v2'][1, :]),
            lambda: self._pbasis[i](v['v3'][0, :], v['v3'][1, :]),
            lambda: self._pbasisdx[i](v['e1'][0, :], v['e1'][1, :])*v['n1'][0, :]
                  + self._pbasisdy[i](v['e1'][0, :], v['e1'][1, :])*v['n1'][1, :],
            lambda: self._pbasisdx[i](v['e2'][0, :], v['e2'][1, :])*v['n2'][0, :]
                  + self._pbasisdy[i](v['e2'][0, :], v['e2'][1, :])*v['n2'][1, :],
            lambda: self._pbasisdx[i](v['e3'][0, :], v['e3'][1, :])*v['n3'][0, :]
                  + self._pbasisdy[i](v['e3'][0, :], v['e3'][1, :])*v['n3'][1, :],
            ][j]()


class ElementGlobalArgyris(ElementGlobal):
    """Argyris element for fourth-order problems."""

    n_dofs = 6
    f_dofs = 1
    dim = 2
    maxdeg = 5

    def gdof(self,v,i,j):
        return [
                # vertex 1
                lambda: self._pbasis[i](v['v1'][0, :], v['v1'][1, :]),
                lambda: self._pbasisdx[i](v['v1'][0, :], v['v1'][1, :]),
                lambda: self._pbasisdy[i](v['v1'][0, :], v['v1'][1, :]),
                lambda: self._pbasisdxx[i](v['v1'][0, :], v['v1'][1, :]),
                lambda: self._pbasisdxy[i](v['v1'][0, :], v['v1'][1, :]),
                lambda: self._pbasisdyy[i](v['v1'][0, :], v['v1'][1, :]),
                # vertex 2
                lambda: self._pbasis[i](v['v2'][0, :], v['v2'][1, :]),
                lambda: self._pbasisdx[i](v['v2'][0, :], v['v2'][1, :]),
                lambda: self._pbasisdy[i](v['v2'][0, :], v['v2'][1, :]),
                lambda: self._pbasisdxx[i](v['v2'][0, :], v['v2'][1, :]),
                lambda: self._pbasisdxy[i](v['v2'][0, :], v['v2'][1, :]),
                lambda: self._pbasisdyy[i](v['v2'][0, :], v['v2'][1, :]),
                # vertex 3
                lambda: self._pbasis[i](v['v3'][0, :], v['v3'][1, :]),
                lambda: self._pbasisdx[i](v['v3'][0, :], v['v3'][1, :]),
                lambda: self._pbasisdy[i](v['v3'][0, :], v['v3'][1, :]),
                lambda: self._pbasisdxx[i](v['v3'][0, :], v['v3'][1, :]),
                lambda: self._pbasisdxy[i](v['v3'][0, :], v['v3'][1, :]),
                lambda: self._pbasisdyy[i](v['v3'][0, :], v['v3'][1, :]),
                # edges in the order 1,2,3
                lambda: self._pbasisdx[i](v['e1'][0, :], v['e1'][1, :])*v['n1'][0, :]
                      + self._pbasisdy[i](v['e1'][0, :], v['e1'][1, :])*v['n1'][1, :],
                lambda: self._pbasisdx[i](v['e2'][0, :], v['e2'][1, :])*v['n2'][0, :]
                      + self._pbasisdy[i](v['e2'][0, :], v['e2'][1, :])*v['n2'][1, :],
                lambda: self._pbasisdx[i](v['e3'][0, :], v['e3'][1, :])*v['n3'][0, :]
                      + self._pbasisdy[i](v['e3'][0, :], v['e3'][1, :])*v['n3'][1, :],
               ][j]()
                

class ElementLocalHdiv(ElementLocal):
    """Abstract :math:`H_{div}` conforming finite element."""

    def gbasis(self,mapping,X,i,tind):
        if isinstance(X,dict):
            raise NotImplementedError("Calling ElementHdiv gbasis with dict not implemented!")
        else:
            dim = X.shape[0]
            x = {}
            x[0] = X[0, :]
            x[1] = X[1, :]
            if dim == 3:
                x[2] = X[2, :]
            [phi, dphi] = self.lbasis(x, i)

        DF = mapping.DF(X, tind)
        detDF = mapping.detDF(X, tind)

        u = {}
        if dim == 2:
            u[0] = (DF[0][0]*phi[0] + DF[0][1]*phi[1])/detDF
            u[1] = (DF[1][0]*phi[0] + DF[1][1]*phi[1])/detDF
        elif dim == 3:
            u[0] = (DF[0][0] * phi[0] + DF[0][1] * phi[1] + DF[0][2] * phi[2]) / detDF
            u[1] = (DF[1][0] * phi[0] + DF[1][1] * phi[1] + DF[1][2] * phi[2]) / detDF
            u[2] = (DF[2][0] * phi[0] + DF[2][1] * phi[1] + DF[2][2] * phi[2]) / detDF

        du = dphi/detDF

        return u, du

class ElementTetRT0(ElementLocalHdiv):
    """Lowest order Raviart-Thomas element for tetrahedron."""
    pass #TODO

class ElementTriRT0(ElementLocalHdiv):
    """Lowest order Raviart-Thomas element for triangle."""

    maxdeg=1
    f_dofs=1
    dim=2

    def lbasis(self,X,i):
        phi={}
        phi[0]={
                0:lambda x,y: x,
                1:lambda x,y: x,
                2:lambda x,y: -x+1.,
                }[i](X[0],X[1])
        phi[1]={
                0:lambda x,y: y-1.,
                1:lambda x,y: y,
                2:lambda x,y: -y,
                }[i](X[0],X[1])
        dphi={
            0:lambda x,y: 2+0.*x,
            1:lambda x,y: 2+0.*x,
            2:lambda x,y: -2+0.*x,
            }[i](X[0],X[1])

        return phi,dphi

class ElementLocalH1(ElementLocal):
    """Abstract :math:`H^1` conforming finite element."""

    def gbasis(self,mapping,X,i,tind):
        if isinstance(X,dict):
            [phi,dphi]=self.lbasis(X,i)
            u=phi
            du={}
        else:
            x={}
            x[0]=X[0,:]
            if mapping.dim>=2:
                x[1]=X[1,:]
            if mapping.dim>=3:
                x[2]=X[2,:]
            [phi,dphi]=self.lbasis(x,i)
            u=np.tile(phi,(len(tind),1))
            du={}
            
        invDF=mapping.invDF(X,tind) # investigate if 'x' should used after else

        self.dim=mapping.dim

        if mapping.dim==1:
            du=np.outer(invDF,dphi)
        elif mapping.dim==2:
            du[0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]
            du[1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]
        elif mapping.dim==3:
            du[0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]+invDF[2][0]*dphi[2]
            du[1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]+invDF[2][1]*dphi[2]
            du[2]=invDF[0][2]*dphi[0]+invDF[1][2]*dphi[1]+invDF[2][2]*dphi[2]
        else:
            raise NotImplementedError("ElementH1.gbasis: not implemented for the given dim.")

        return u,du

class ElementLocalH1Vec(ElementLocalH1):
    """Convert :math:`H^1` element to vectorial :math:`H^1` element."""
    def __init__(self,elem):
        if elem.dim==0:
            print("ElementH1Vec.__init__(): Warning! Parent element has no dim-variable!")
        self.dim=elem.dim
        self.elem=elem
        # multiplicate the amount of DOF's with dim
        self.n_dofs=self.elem.n_dofs*self.dim
        self.f_dofs=self.elem.f_dofs*self.dim
        self.i_dofs=self.elem.i_dofs*self.dim
        self.e_dofs=self.elem.e_dofs*self.dim
        self.maxdeg=elem.maxdeg

    def gbasis(self,mapping,X,i,tind):
        ind=np.floor(float(i)/float(self.dim))
        n=i-self.dim*ind

        u={}
        du={}

        if isinstance(X,dict):
            [phi,dphi]=self.elem.lbasis(X,ind)
        else:
            x={}
            x[0]=X[0,:]
            x[1]=X[1,:]
            if mapping.dim>=3:
                x[2]=X[2,:]
            [phi,dphi]=self.elem.lbasis(x,ind)
            phi=np.tile(phi,(len(tind),1))
            for itr in range(self.dim):
                dphi[itr]=np.tile(dphi[itr],(len(tind),1))

        # fill appropriate slots of u and du (u[0] -> x-component of u etc.)
        for itr in range(self.dim):
            if itr==n:
                u[itr]=phi
                invDF=mapping.invDF(X,tind) # investigate if 'x' should used after else
                du[itr]={}
                if mapping.dim==2:
                    du[itr][0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]
                    du[itr][1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]
                elif mapping.dim==3:
                    du[itr][0]=invDF[0][0]*dphi[0]+invDF[1][0]*dphi[1]+invDF[2][0]*dphi[2]
                    du[itr][1]=invDF[0][1]*dphi[0]+invDF[1][1]*dphi[1]+invDF[2][1]*dphi[2]
                    du[itr][2]=invDF[0][2]*dphi[0]+invDF[1][2]*dphi[1]+invDF[2][2]*dphi[2]
                else:
                    raise NotImplementedError("ElementH1Vec.gbasis: not implemented for the given dim.")
            else:
                u[itr]=0*phi
                du[itr]={}
                for jtr in range(self.dim):
                    du[itr][jtr]=0*phi
            
        return u,du
        
class ElementLocalQ1(ElementLocalH1):
    """Simplest quadrilateral element."""
    
    maxdeg=2
    n_dofs=1
    dim=2
        
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 0.25*(1-x)*(1-y),
            1:lambda x,y: 0.25*(1+x)*(1-y),
            2:lambda x,y: 0.25*(1+x)*(1+y),
            3:lambda x,y: 0.25*(1-x)*(1+y)
            }[i](X[0],X[1])
        dphi={}
        dphi[0]={
            0:lambda x,y: 0.25*(-1+y),
            1:lambda x,y: 0.25*(1-y),
            2:lambda x,y: 0.25*(1+y),
            3:lambda x,y: 0.25*(-1-y)
            }[i](X[0],X[1])
        dphi[1]={
            0:lambda x,y: 0.25*(-1+x),
            1:lambda x,y: 0.25*(-1-x),
            2:lambda x,y: 0.25*(1+x),
            3:lambda x,y: 0.25*(1-x)
            }[i](X[0],X[1])
        return phi,dphi
        
class ElementLocalQ2(ElementLocalH1):
    """Second order quadrilateral element."""

    maxdeg=3
    n_dofs=1
    f_dofs=1
    i_dofs=1
    dim=2
        
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 0.25*(x**2-x)*(y**2-y),
            1:lambda x,y: 0.25*(x**2+x)*(y**2-y),
            2:lambda x,y: 0.25*(x**2+x)*(y**2+y),
            3:lambda x,y: 0.25*(x**2-x)*(y**2+y),
            4:lambda x,y: 0.5*(y**2-y)*(1-x**2),
            5:lambda x,y: 0.5*(x**2+x)*(1-y**2),
            6:lambda x,y: 0.5*(y**2+y)*(1-x**2),
            7:lambda x,y: 0.5*(x**2-x)*(1-y**2),
            8:lambda x,y: (1-x**2)*(1-y**2)
            }[i](X[0],X[1])
        dphi={}
        dphi[0]={
            0:lambda x,y:((-1 + 2*x)*(-1 + y)*y)/4.,
            1:lambda x,y:((1 + 2*x)*(-1 + y)*y)/4.,
            2:lambda x,y:((1 + 2*x)*y*(1 + y))/4.,
            3:lambda x,y:((-1 + 2*x)*y*(1 + y))/4.,
            4:lambda x,y:-(x*(-1 + y)*y),
            5:lambda x,y:-((1 + 2*x)*(-1 + y**2))/2.,
            6:lambda x,y:-(x*y*(1 + y)),
            7:lambda x,y:-((-1 + 2*x)*(-1 + y**2))/2.,
            8:lambda x,y:2*x*(-1 + y**2)
            }[i](X[0],X[1])
        dphi[1]={
            0:lambda x,y:((-1 + x)*x*(-1 + 2*y))/4.,
            1:lambda x,y:(x*(1 + x)*(-1 + 2*y))/4.,
            2:lambda x,y:(x*(1 + x)*(1 + 2*y))/4.,
            3:lambda x,y:((-1 + x)*x*(1 + 2*y))/4.,
            4:lambda x,y:-((-1 + x**2)*(-1 + 2*y))/2.,
            5:lambda x,y:-(x*(1 + x)*y),
            6:lambda x,y:-((-1 + x**2)*(1 + 2*y))/2.,
            7:lambda x,y:-((-1 + x)*x*y),
            8:lambda x,y:2*(-1 + x**2)*y
            }[i](X[0],X[1])
        return phi,dphi

class ElementLocalTriPp(ElementLocalH1):
    """A somewhat slow implementation of hierarchical
    p-basis for triangular mesh."""

    dim=2

    def __init__(self,p):
        self.p=p
        self.maxdeg=p
        self.n_dofs=1
        self.f_dofs=np.max([p-1,0])
        self.i_dofs=np.max([(p-1)*(p-2)/2,0])

        self.nbdofs=3*self.n_dofs+3*self.f_dofs+self.i_dofs

    def intlegpoly(self,x,n):
        # Generate integrated Legendre polynomials.
        n=n+1
        
        P={}
        P[0]=np.ones(x.shape)
        if n>1:
            P[1]=x
        
        for i in np.arange(1,n):
            P[i+1]=((2.*i+1.)/(i+1.))*x*P[i]-(i/(i+1.))*P[i-1]
            
        iP={}
        iP[0]=np.ones(x.shape)
        if n>1:
            iP[1]=x
            
        for i in np.arange(1,n-1):
            iP[i+1]=(P[i+1]-P[i-1])/(2.*i+1.)
            
        dP={}
        dP[0]=np.zeros(x.shape)
        for i in np.arange(1,n):
            dP[i]=P[i-1]
        
        return iP,dP
        
    def lbasis(self,X,n):
        # Evaluate n'th Lagrange basis of order self.p.
        p=self.p

        if len(X)!=2:
            raise NotImplementedError("ElementTriPp: not implemented for the given dimension of X.")

        phi={}
        phi[0]=1.-X[0]-X[1]
        phi[1]=X[0]
        phi[2]=X[1]
        
        # local basis function gradients TODO fix these somehow
        gradphi_x={}
        gradphi_x[0]=-1.*np.ones(X[0].shape)
        gradphi_x[1]=1.*np.ones(X[0].shape)
        gradphi_x[2]=np.zeros(X[0].shape)
        
        gradphi_y={}
        gradphi_y[0]=-1.*np.ones(X[0].shape)
        gradphi_y[1]=np.zeros(X[0].shape)
        gradphi_y[2]=1.*np.ones(X[0].shape)

        if n<=2:
            # return first three
            dphi={}
            dphi[0]=gradphi_x[n]
            dphi[1]=gradphi_y[n]
            return phi[n],dphi

        # use same ordering as in mesh
        e=np.array([[0,1],[1,2],[0,2]]).T
        offset=3
        
        # define edge basis functions
        if(p>1):
            for i in range(3):
                eta=phi[e[1,i]]-phi[e[0,i]]
                deta_x=gradphi_x[e[1,i]]-gradphi_x[e[0,i]]
                deta_y=gradphi_y[e[1,i]]-gradphi_y[e[0,i]]
                
                # generate integrated Legendre polynomials
                [P,dP]=self.intlegpoly(eta,p-2)
                
                for j in range(len(P)):
                    phi[offset]=phi[e[0,i]]*phi[e[1,i]]*P[j]
                    gradphi_x[offset]=gradphi_x[e[0,i]]*phi[e[1,i]]*P[j]+\
                                      gradphi_x[e[1,i]]*phi[e[0,i]]*P[j]+\
                                      deta_x*phi[e[0,i]]*phi[e[1,i]]*dP[j]
                    gradphi_y[offset]=gradphi_y[e[0,i]]*phi[e[1,i]]*P[j]+\
                                      gradphi_y[e[1,i]]*phi[e[0,i]]*P[j]+\
                                      deta_y*phi[e[0,i]]*phi[e[1,i]]*dP[j]
                    if offset==n:
                        # return if computed
                        dphi={}
                        dphi[0]=gradphi_x[n]
                        dphi[1]=gradphi_y[n]
                        return phi[n],dphi
                    offset=offset+1  
        
        # define interior basis functions
        if(p>2):
            B={}
            dB_x={}
            dB_y={}
            if(p>3):
                pm3=ElementLocalTriPp(p - 3)
                for itr in range(pm3.nbdofs):
                    pphi,pdphi=self.lbasis(X,itr)
                    B[itr]=pphi
                    dB_x[itr]=pdphi[0]
                    dB_y[itr]=pdphi[1]
                N=pm3.nbdofs
            else:
                B[0]=np.ones(X[0].shape)
                dB_x[0]=np.zeros(X[0].shape)
                dB_y[0]=np.zeros(X[0].shape)
                N=1
                
            bubble=phi[0]*phi[1]*phi[2]
            dbubble_x=gradphi_x[0]*phi[1]*phi[2]+\
                      gradphi_x[1]*phi[2]*phi[0]+\
                      gradphi_x[2]*phi[0]*phi[1]
            dbubble_y=gradphi_y[0]*phi[1]*phi[2]+\
                      gradphi_y[1]*phi[2]*phi[0]+\
                      gradphi_y[2]*phi[0]*phi[1]
            
            for i in range(N):
                phi[offset]=bubble*B[i]
                gradphi_x[offset]=dbubble_x*B[i]+dB_x[i]*bubble
                gradphi_y[offset]=dbubble_y*B[i]+dB_y[i]*bubble
                if offset==n:
                    # return if computed
                    dphi={}
                    dphi[0]=gradphi_x[n]
                    dphi[1]=gradphi_y[n]
                    return phi[n],dphi
                offset=offset+1

        raise IndexError("ElementTriPp.lbasis: reached end of lbasis without returning anything.")

class ElementLocalTriDG(ElementLocalH1):
    """Transform a H1 conforming triangular element
    into a discontinuous one by turning all DOFs into
    interior DOFs."""
    def __init__(self,elem):
        # change all dofs to interior dofs
        self.elem=elem
        self.maxdeg=elem.maxdeg
        self.i_dofs=3*elem.n_dofs+3*elem.f_dofs+elem.i_dofs
        self.dim=2
    def lbasis(self,X,i):
        return self.elem.lbasis(X,i)

class ElementLocalTetDG(ElementLocalH1):
    """Convert a H1 tetrahedral element into a DG element.
    All DOFs are converted to interior DOFs."""
    def __init__(self,elem):
        # change all dofs to interior dofs
        self.elem=elem
        self.maxdeg=elem.maxdeg
        self.i_dofs=4*elem.n_dofs+4*elem.f_dofs+6*elem.e_dofs+elem.i_dofs
        self.dim=3
    def lbasis(self,X,i):
        return self.elem.lbasis(X,i)
    
class ElementLocalTetP0(ElementLocalH1):
    """Piecewise constant element for tetrahedral mesh."""

    i_dofs=1
    maxdeg=1
    dim=3

    def lbasis(self,X,i):
        phi={
            0:lambda x,y,z: 1+0*x
            }[i](X[0],X[1],X[2])
        dphi={}
        dphi[0]={
                0:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[1]={
                0:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[2]={
                0:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        return phi,dphi

class ElementLocalTriP0(ElementLocalH1):
    """Piecewise constant element for triangular mesh."""

    i_dofs=1
    maxdeg=1
    dim=2

    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1+0*x
            }[i](X[0],X[1])
        dphi={}
        dphi[0]={
                0:lambda x,y: 0*x
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: 0*x
                }[i](X[0],X[1])
        return phi,dphi

# this is for legacy
class ElementLocalP0(ElementLocalTriP0):
    pass

class ElementLocalTriMini(ElementLocalH1):
    """The MINI-element for triangular mesh."""

    dim=2
    n_dofs=1
    i_dofs=1
    maxdeg=3

    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1-x-y,
            1:lambda x,y: x,
            2:lambda x,y: y,
            3:lambda x,y: (1-x-y)*x*y
            }[i](X[0],X[1])

        dphi={}
        dphi[0]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 1+0*x,
                2:lambda x,y: 0*x,
                3:lambda x,y: (1-x-y)*y-x*y
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 0*x,
                2:lambda x,y: 1+0*x,
                3:lambda x,y: (1-x-y)*x-x*y
                }[i](X[0],X[1])
        return phi,dphi
        
class ElementLocalTetP2(ElementLocalH1):
    """The quadratic tetrahedral element."""
    
    dim=3
    n_dofs=1
    e_dofs=1
    maxdeg=2
    
    def lbasis(self,X,i):
        
        phi={ # order (0,0,0) (1,0,0) (0,1,0) (0,0,1) and then according to mesh local t2e
            0:lambda x,y,z: 1. - 3.*x + 2.*x**2 - 3.*y + 4.*x*y + 2.*y**2 - 3.*z + 4.*x*z + 4.*y*z + 2.*z**2,
            1:lambda x,y,z: 0. - 1.*x + 2.*x**2,
            2:lambda x,y,z: 0. - 1.*y + 2.*y**2,
            3:lambda x,y,z: 0. - 1.*z + 2.*z**2,
            4:lambda x,y,z: 0. + 4.*x - 4.*x**2 - 4.*x*y - 4.*x*z,
            5:lambda x,y,z: 0. + 4.*x*y,
            6:lambda x,y,z: 0. + 4.*y - 4.*x*y - 4.*y**2 - 4.*y*z,
            7:lambda x,y,z: 0. + 4.*z - 4.*x*z - 4.*y*z - 4.*z**2,
            8:lambda x,y,z: 0. + 4.*x*z,
            9:lambda x,y,z: 0. + 4.*y*z,
            }[i](X[0],X[1],X[2])
    
        dphi={}
        dphi[0]={
                0:lambda x,y,z: -3. + 4.*x + 4.*y + 4.*z,
                1:lambda x,y,z: -1. + 4.*x,
                2:lambda x,y,z: 0.*x,
                3:lambda x,y,z: 0.*x,
                4:lambda x,y,z: 4. - 8.*x - 4.*y - 4.*z,
                5:lambda x,y,z: 4.*y,
                6:lambda x,y,z: -4.*y,
                7:lambda x,y,z: -4.*z,
                8:lambda x,y,z: 4.*z,
                9:lambda x,y,z: 0.*x,
                }[i](X[0],X[1],X[2])
        dphi[1]={
                0:lambda x,y,z: -3. + 4.*x + 4.*y + 4.*z,
                1:lambda x,y,z: 0.*x,
                2:lambda x,y,z: -1. + 4.*y,
                3:lambda x,y,z: 0.*x,
                4:lambda x,y,z: -4.*x,
                5:lambda x,y,z: 4.*x,
                6:lambda x,y,z: 4. - 4.*x - 8.*y - 4.*z,
                7:lambda x,y,z: -4.*z,
                8:lambda x,y,z: 0.*x,
                9:lambda x,y,z: 4.*z,
                }[i](X[0],X[1],X[2])
        dphi[2]={
                0:lambda x,y,z: -3. + 4.*x + 4.*y + 4.*z,
                1:lambda x,y,z: 0.*x,
                2:lambda x,y,z: 0.*x,
                3:lambda x,y,z: -1. + 4.*z,
                4:lambda x,y,z: -4.*x,
                5:lambda x,y,z: 0.*x,
                6:lambda x,y,z: -4.*y,
                7:lambda x,y,z: 4. - 4.*x - 4.*y - 8.*z,
                8:lambda x,y,z: 4.*x,
                9:lambda x,y,z: 4.*y,       
                }[i](X[0],X[1],X[2])
                
        return phi,dphi

class ElementLocalLineP2(ElementLocalH1):
    """Quadratic element for one dimension."""

    n_dofs = 1
    i_dofs = 1
    dim = 1
    maxdeg = 2
    
    def lbasis(self, X, i):
        phi = {
            0: lambda x: 1-x,
            1: lambda x: x,
            2: lambda x: 4*x-4*x**2,
            }[i](X[0])

        dphi={}
        dphi={
                0: lambda x: -1+0*x,
                1: lambda x: 1+0*x,
                2: lambda x: 4-8*x,
                }[i](X[0])
                
        return phi, dphi        
        
class ElementLocalLineP1(ElementLocalH1):
    """Linear element for one dimension."""

    n_dofs = 1
    dim = 1
    maxdeg = 1
    
    def lbasis(self, X, i):
        phi = {
            0: lambda x: 1-x,
            1: lambda x: x,
            }[i](X[0])

        dphi={}
        dphi={
                0: lambda x: -1+0*x,
                1: lambda x: 1+0*x,
                }[i](X[0])
                
        return phi, dphi        
        
class ElementLocalTriP1(ElementLocalH1):
    """The simplest triangular element."""

    n_dofs=1
    dim=2
    maxdeg=1
    
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1-x-y,
            1:lambda x,y: x,
            2:lambda x,y: y
            }[i](X[0],X[1])

        dphi={}
        dphi[0]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 1+0*x,
                2:lambda x,y: 0*x
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: -1+0*x,
                1:lambda x,y: 0*x,
                2:lambda x,y: 1+0*x
                }[i](X[0],X[1])
                
        return phi,dphi

class ElementLocalTriP2(ElementLocalH1):
    """The quadratic triangular element."""

    n_dofs=1
    f_dofs=1
    dim=2
    maxdeg=2
    
    def lbasis(self,X,i):
        phi={
            0:lambda x,y: 1-3*x-3*y+2*x**2+4*x*y+2*y**2,
            1:lambda x,y: 2*x**2-x,
            2:lambda x,y: 2*y**2-y,
            3:lambda x,y: 4*x-4*x**2-4*x*y,
            4:lambda x,y: 4*x*y,
            5:lambda x,y: 4*y-4*x*y-4*y**2,
            }[i](X[0],X[1])

        dphi={}
        dphi[0]={
                0:lambda x,y: -3+4*x+4*y,
                1:lambda x,y: 4*x-1,
                2:lambda x,y: 0*x,
                3:lambda x,y: 4-8*x-4*y,
                4:lambda x,y: 4*y,
                5:lambda x,y: -4*y,
                }[i](X[0],X[1])
        dphi[1]={
                0:lambda x,y: -3+4*x+4*y,
                1:lambda x,y: 0*x,
                2:lambda x,y: 4*y-1,
                3:lambda x,y: -4*x,
                4:lambda x,y: 4*x,
                5:lambda x,y: 4-4*x-8*y,
                }[i](X[0],X[1])
                
        return phi,dphi
        
class ElementLocalTetP1(ElementLocalH1):
    """The simplest tetrahedral element."""
    
    n_dofs=1
    maxdeg=1
    dim=3

    def lbasis(self,X,i):

        phi={
            0:lambda x,y,z: 1-x-y-z,
            1:lambda x,y,z: x,
            2:lambda x,y,z: y,
            3:lambda x,y,z: z,
            }[i](X[0],X[1],X[2])

        dphi={}
        dphi[0]={
                0:lambda x,y,z: -1+0*x,
                1:lambda x,y,z: 1+0*x,
                2:lambda x,y,z: 0*x,
                3:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[1]={
                0:lambda x,y,z: -1+0*x,
                1:lambda x,y,z: 0*x,
                2:lambda x,y,z: 1+0*x,
                3:lambda x,y,z: 0*x
                }[i](X[0],X[1],X[2])
        dphi[2]={
                0:lambda x,y,z: -1+0*x,
                1:lambda x,y,z: 0*x,
                2:lambda x,y,z: 0*x,
                3:lambda x,y,z: 1+0*x
                }[i](X[0],X[1],X[2])

        return phi,dphi
