# -*- coding: utf-8 -*-
"""
The mappings defining relationships between reference and global elements.

Currently these classes have quite a lot of undocumented behavior and
untested code. The following mappings are implemented to some extent:

    * :class:`skfem.mapping.MappingAffine`, the standard affine local-to-global mapping that can be used with triangular and tetrahedral elements.
    * :class:`skfem.mapping.MappingQ1`, the local-to-global mapping defined by the Q1 basis functions. This is required for quadrilateral meshes.
"""
import numpy as np
import copy

class Mapping:
    """Abstract class for mappings."""
    dim=0

    def __init__(self,mesh):
        raise NotImplementedError("Constructor not implemented!")

    def F(self,X,tind):
        """Element local to global."""
        raise NotImplementedError("F() not implemented!")

    def invF(self,x,tind):
        raise NotImplementedError("invF() not implemented!")

    def DF(self,X,tind):
        raise NotImplementedError("DF() not implemented!")

    def invDF(self,X,tind):
        raise NotImplementedError("invDF() not implemented!")

    def detDF(self,X,tind):
        raise NotImplementedError("detDF() not implemented!")

    def G(self,X,find):
        """Boundary local to global."""
        raise NotImplementedError("G() not implemented!")
        
    def detDG(self,X,find):
        raise NotImplementedError("detDG() not implemented!")

    def normals(self,X,find):
        raise NotImplementedError("normals() not implemented!")

class MappingQ1(Mapping):
    """Mapping for quadrilaterals."""
    
    def __init__(self,mesh):
        import skfem.mesh as fmsh
        if isinstance(mesh,fmsh.MeshQuad):
            self.dim=2
            
            self.t=mesh.t
            self.p=mesh.p
            
            self.J={0:{},1:{}}
            self.J[0][0]=lambda x,y,t:0.25*(-mesh.p[0,mesh.t[0,t]][:,None]*(1-y)\
                                            +mesh.p[0,mesh.t[1,t]][:,None]*(1-y)\
                                            +mesh.p[0,mesh.t[2,t]][:,None]*(1+y)\
                                            -mesh.p[0,mesh.t[3,t]][:,None]*(1+y))
            self.J[0][1]=lambda x,y,t:0.25*(-mesh.p[0,mesh.t[0,t]][:,None]*(1-x)\
                                            -mesh.p[0,mesh.t[1,t]][:,None]*(1+x)\
                                            +mesh.p[0,mesh.t[2,t]][:,None]*(1+x)\
                                            +mesh.p[0,mesh.t[3,t]][:,None]*(1-x))
            self.J[1][0]=lambda x,y,t:0.25*(-mesh.p[1,mesh.t[0,t]][:,None]*(1-y)\
                                            +mesh.p[1,mesh.t[1,t]][:,None]*(1-y)\
                                            +mesh.p[1,mesh.t[2,t]][:,None]*(1+y)\
                                            -mesh.p[1,mesh.t[3,t]][:,None]*(1+y))
            self.J[1][1]=lambda x,y,t:0.25*(-mesh.p[1,mesh.t[0,t]][:,None]*(1-x)\
                                            -mesh.p[1,mesh.t[1,t]][:,None]*(1+x)\
                                            +mesh.p[1,mesh.t[2,t]][:,None]*(1+x)\
                                            +mesh.p[1,mesh.t[3,t]][:,None]*(1-x))
                                          
            # Matrices and vectors for boundary mappings: G(X)=BX+c
            self.B={}
    
            self.B[0]=mesh.p[0,mesh.facets[1,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[1]=mesh.p[1,mesh.facets[1,:]]-mesh.p[1,mesh.facets[0,:]]
    
            self.c={}
    
            self.c[0]=mesh.p[0,mesh.facets[0,:]]
            self.c[1]=mesh.p[1,mesh.facets[0,:]]
    
            self.detB=np.sqrt(self.B[0]**2+self.B[1]**2)
        else:
            raise NotImplementedError("MappingQ1: wrong type of mesh was given to constructor!")

    def quadbasis(self,x,y,i):
        return {
            0:lambda x,y: 0.25*(1-x)*(1-y),
            1:lambda x,y: 0.25*(1+x)*(1-y),
            2:lambda x,y: 0.25*(1+x)*(1+y),
            3:lambda x,y: 0.25*(1-x)*(1+y)
            }[i](x,y)
   
    def F(self,Y,tind=None):
        """Mapping defined by Q1 basis."""
        out={}

        if not isinstance(Y,dict):
            X={}
            X[0]=Y[0,:]
            X[1]=Y[1,:]
        else:
            X=Y
            
        if tind is None:
            tind=range(self.t.shape[1])

        out[0]=self.p[0,self.t[0,tind]][:,None]*self.quadbasis(X[0],X[1],0)+\
               self.p[0,self.t[1,tind]][:,None]*self.quadbasis(X[0],X[1],1)+\
               self.p[0,self.t[2,tind]][:,None]*self.quadbasis(X[0],X[1],2)+\
               self.p[0,self.t[3,tind]][:,None]*self.quadbasis(X[0],X[1],3)
        out[1]=self.p[1,self.t[0,tind]][:,None]*self.quadbasis(X[0],X[1],0)+\
               self.p[1,self.t[1,tind]][:,None]*self.quadbasis(X[0],X[1],1)+\
               self.p[1,self.t[2,tind]][:,None]*self.quadbasis(X[0],X[1],2)+\
               self.p[1,self.t[3,tind]][:,None]*self.quadbasis(X[0],X[1],3)     
        
        return out
        
    def invF(self,x,tind=None):
        """Inverse map. Perform Newton iteration."""
        X={}
        X[0]=0*x[0]
        X[1]=0*x[1]
        for itr in range(1): # One Newton iteration. Should be enough?
            g={}
            F=self.F(X,tind)
            invDF=self.invDF(X,tind)
            g[0]=x[0]-F[0]
            g[1]=x[1]-F[1]
            xnext={}
            xnext[0]=X[0]+invDF[0][0]*g[0]+invDF[0][1]*g[1]
            xnext[1]=X[1]+invDF[1][0]*g[0]+invDF[1][1]*g[1]
            
        return xnext
            
        
    def detDF(self,X,tind=None):
        if tind is None:
            tind=range(self.t.shape[1])
        if isinstance(X,dict):
            detDF=self.J[0][0](X[0],X[1],tind)*self.J[1][1](X[0],X[1],tind)-\
                  self.J[0][1](X[0],X[1],tind)*self.J[1][0](X[0],X[1],tind) 
        else:
            detDF=self.J[0][0](X[0,:],X[1,:],tind)*self.J[1][1](X[0,:],X[1,:],tind)-\
                  self.J[0][1](X[0,:],X[1,:],tind)*self.J[1][0](X[0,:],X[1,:],tind)
        return detDF
            
    def invDF(self,X,tind=None):
        invJ={0:{},1:{}}
        if isinstance(X,dict):
            x=X[0]
            y=X[1]
        else:
            x=X[0,:]
            y=X[1,:]
            
        if tind is None:
            tind=range(self.t.shape[1])
                
        detDF=self.detDF(X,tind)
        invJ[0][0]=self.J[1][1](x,y,tind)/detDF
        invJ[0][1]=-self.J[0][1](x,y,tind)/detDF
        invJ[1][0]=-self.J[1][0](x,y,tind)/detDF
        invJ[1][1]=self.J[0][0](x,y,tind)/detDF
        
        return invJ

    def normals(self,X,tind,find,t2f):
        N={}

        nref=np.array([[0.0,-1.0],[1.0,0.0],[0.0,1.0],[-1.0,0.0]])

        invDF=self.invDF(X,tind)

        # initialize n to zero
        n={}
        n[0]=np.zeros(find.shape[0]) # of size Nfacets x Nqp
        n[1]=np.zeros(find.shape[0])

        # compute all local normals
        for itr in range(nref.shape[0]):
            inds=np.nonzero(t2f[itr,tind]==find)[0]
            for jtr in range(nref.shape[1]):
               n[jtr][inds]=nref[itr,jtr]

        # map to global normals
        N[0]=invDF[0][0]*n[0][:,None]+invDF[1][0]*n[1][:,None]
        N[1]=invDF[0][1]*n[0][:,None]+invDF[1][1]*n[1][:,None]

        # normalize
        nlen=np.sqrt(N[0]**2+N[1]**2)

        # shrink to required facets and tile
        N[0]=N[0]/nlen
        N[1]=N[1]/nlen

        return N # n[0] etc. are of size Nfacets x Nqp
        
    def G(self,X,find=None):
        """Boundary mapping :math:`G(X)=BX+c`."""
        y={}
        if find is None:
            y[0]=np.outer(self.B[0],X).T+self.c[0]
            y[1]=np.outer(self.B[1],X).T+self.c[1]
        else:
            y[0]=np.outer(self.B[0][find],X).T+self.c[0][find]
            y[1]=np.outer(self.B[1][find],X).T+self.c[1][find]
        y[0]=y[0].T
        y[1]=y[1].T
        return y
        
    def detDG(self,X,find=None):
        if find is None:
            detDG=self.detB
        else:
            detDG=self.detB[find]
        return np.tile(detDG,(X.shape[1],1)).T

class MappingAffineMortar(Mapping):
    """Affine mappings for simplex mortar meshes."""
    def __init__(self, mesh):
        self.B={}

        self.B[0] = mesh.p[0, mesh.facets[1, :]] - mesh.p[0, mesh.facets[0, :]]
        self.B[1] = mesh.p[1, mesh.facets[1, :]] - mesh.p[1, mesh.facets[0, :]]

        self.c={}

        self.c[0] = mesh.p[0, mesh.facets[0, :]]
        self.c[1] = mesh.p[1, mesh.facets[0, :]]

        self.detB=np.sqrt(self.B[0]**2+self.B[1]**2)

    def G(self, X, find=None):
        """Mortar mapping G(X)=Bx+c."""
        y={}

        if find is None:
            y[0] = np.outer(self.B[0],X).T+self.c[0]
            y[1] = np.outer(self.B[1],X).T+self.c[1]
        else:
            y[0] = np.outer(self.B[0][find],X).T+self.c[0][find]
            y[1] = np.outer(self.B[1][find],X).T+self.c[1][find]
        y[0] = y[0].T
        y[1] = y[1].T

        return y

    def detDG(self, X, find=None):
        if find is None:
            detDG=self.detB
        else:
            detDG=self.detB[find]
        return np.tile(detDG,(X.shape[1],1)).T

class MappingAffine(Mapping):
    """Affine mappings for simplex (=line,tri,tet) mesh."""
    def __init__(self,mesh):
        import skfem.mesh as fmsh
        if isinstance(mesh,fmsh.MeshLine):
            self.dim=1
            
            self.A=mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
            self.b=mesh.p[0,mesh.t[0,:]]
            
            self.detA=self.A
            
            self.invA=1.0/self.A
          
        elif isinstance(mesh,fmsh.MeshTri):
            self.dim=2            
            
            self.A={0:{},1:{}}
    
            self.A[0][0]=mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[0][1]=mesh.p[0,mesh.t[2,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[1][0]=mesh.p[1,mesh.t[1,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[1][1]=mesh.p[1,mesh.t[2,:]]-mesh.p[1,mesh.t[0,:]]
    
            self.b={}
    
            self.b[0]=mesh.p[0,mesh.t[0,:]]
            self.b[1]=mesh.p[1,mesh.t[0,:]]
    
            self.detA=self.A[0][0]*self.A[1][1]-self.A[0][1]*self.A[1][0]
    
            self.invA={0:{},1:{}}
    
            self.invA[0][0]=self.A[1][1]/self.detA
            self.invA[0][1]=-self.A[0][1]/self.detA
            self.invA[1][0]=-self.A[1][0]/self.detA
            self.invA[1][1]=self.A[0][0]/self.detA 
    
            # Matrices and vectors for boundary mappings: G(X)=BX+c
            self.B={}
    
            self.B[0]=mesh.p[0,mesh.facets[1,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[1]=mesh.p[1,mesh.facets[1,:]]-mesh.p[1,mesh.facets[0,:]]
    
            self.c={}
    
            self.c[0]=mesh.p[0,mesh.facets[0,:]]
            self.c[1]=mesh.p[1,mesh.facets[0,:]]
    
            self.detB=np.sqrt(self.B[0]**2+self.B[1]**2)
            
        elif isinstance(mesh,fmsh.MeshTet):
            self.dim=3            
            
            self.A={0:{},1:{},2:{}}
    
            self.A[0][0]=mesh.p[0,mesh.t[1,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[0][1]=mesh.p[0,mesh.t[2,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[0][2]=mesh.p[0,mesh.t[3,:]]-mesh.p[0,mesh.t[0,:]]
            self.A[1][0]=mesh.p[1,mesh.t[1,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[1][1]=mesh.p[1,mesh.t[2,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[1][2]=mesh.p[1,mesh.t[3,:]]-mesh.p[1,mesh.t[0,:]]
            self.A[2][0]=mesh.p[2,mesh.t[1,:]]-mesh.p[2,mesh.t[0,:]]
            self.A[2][1]=mesh.p[2,mesh.t[2,:]]-mesh.p[2,mesh.t[0,:]]
            self.A[2][2]=mesh.p[2,mesh.t[3,:]]-mesh.p[2,mesh.t[0,:]]
    
            self.b={}
    
            self.b[0]=mesh.p[0,mesh.t[0,:]]
            self.b[1]=mesh.p[1,mesh.t[0,:]]
            self.b[2]=mesh.p[2,mesh.t[0,:]]
    
            self.detA=self.A[0][0]*(self.A[1][1]*self.A[2][2]-self.A[1][2]*self.A[2][1])\
                      -self.A[0][1]*(self.A[1][0]*self.A[2][2]-self.A[1][2]*self.A[2][0])\
                      +self.A[0][2]*(self.A[1][0]*self.A[2][1]-self.A[1][1]*self.A[2][0])
    
            self.invA={0:{},1:{},2:{}}
    
            self.invA[0][0]=(-self.A[1][2]*self.A[2][1]+self.A[1][1]*self.A[2][2])/self.detA
            self.invA[1][0]=( self.A[1][2]*self.A[2][0]-self.A[1][0]*self.A[2][2])/self.detA
            self.invA[2][0]=(-self.A[1][1]*self.A[2][0]+self.A[1][0]*self.A[2][1])/self.detA
            self.invA[0][1]=( self.A[0][2]*self.A[2][1]-self.A[0][1]*self.A[2][2])/self.detA
            self.invA[1][1]=(-self.A[0][2]*self.A[2][0]+self.A[0][0]*self.A[2][2])/self.detA
            self.invA[2][1]=( self.A[0][1]*self.A[2][0]-self.A[0][0]*self.A[2][1])/self.detA
            self.invA[0][2]=(-self.A[0][2]*self.A[1][1]+self.A[0][1]*self.A[1][2])/self.detA
            self.invA[1][2]=( self.A[0][2]*self.A[1][0]-self.A[0][0]*self.A[1][2])/self.detA
            self.invA[2][2]=(-self.A[0][1]*self.A[1][0]+self.A[0][0]*self.A[1][1])/self.detA
    
            # Matrices and vectors for boundary mappings: G(X)=BX+c
            self.B={0:{},1:{},2:{}}
    
            self.B[0][0]=mesh.p[0,mesh.facets[1,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[0][1]=mesh.p[0,mesh.facets[2,:]]-mesh.p[0,mesh.facets[0,:]]
            self.B[1][0]=mesh.p[1,mesh.facets[1,:]]-mesh.p[1,mesh.facets[0,:]]
            self.B[1][1]=mesh.p[1,mesh.facets[2,:]]-mesh.p[1,mesh.facets[0,:]]
            self.B[2][0]=mesh.p[2,mesh.facets[1,:]]-mesh.p[2,mesh.facets[0,:]]
            self.B[2][1]=mesh.p[2,mesh.facets[2,:]]-mesh.p[2,mesh.facets[0,:]]
    
            self.c={}
    
            self.c[0]=mesh.p[0,mesh.facets[0,:]]
            self.c[1]=mesh.p[1,mesh.facets[0,:]]
            self.c[2]=mesh.p[2,mesh.facets[0,:]]
    
            crossp={}
            crossp[0]= self.B[1][0]*self.B[2][1]-self.B[2][0]*self.B[1][1]
            crossp[1]=-self.B[0][0]*self.B[2][1]+self.B[2][0]*self.B[0][1]
            crossp[2]= self.B[0][0]*self.B[1][1]-self.B[1][0]*self.B[0][1]
    
            self.detB=np.sqrt(crossp[0]**2+crossp[1]**2+crossp[2]**2)
            
        else:
            raise TypeError("MappingAffine initialized with an incompatible mesh type!")

    def F(self,X,tind=None):
        """Affine map F(X)=AX+b."""
        y={}
        if self.dim==1:
            y=np.outer(self.A,X[0,:]).T+self.b
            y=y.T
        elif self.dim==2:
            if tind is None:
                y[0]=np.outer(self.A[0][0],X[0,:]).T+np.outer(self.A[0][1],X[1,:]).T+self.b[0]
                y[1]=np.outer(self.A[1][0],X[0,:]).T+np.outer(self.A[1][1],X[1,:]).T+self.b[1]
            else: # TODO check this could have error
                y[0]=np.outer(self.A[0][0][tind],X[0,:]).T+np.outer(self.A[0][1][tind],X[1,:]).T+self.b[0][tind]
                y[1]=np.outer(self.A[1][0][tind],X[0,:]).T+np.outer(self.A[1][1][tind],X[1,:]).T+self.b[1][tind]
            y[0]=y[0].T
            y[1]=y[1].T
        elif self.dim==3:
            if tind is None:
                y[0]=np.outer(self.A[0][0],X[0,:]).T+\
                     np.outer(self.A[0][1],X[1,:]).T+\
                     np.outer(self.A[0][2],X[2,:]).T+self.b[0]
                y[1]=np.outer(self.A[1][0],X[0,:]).T+\
                     np.outer(self.A[1][1],X[1,:]).T+\
                     np.outer(self.A[1][2],X[2,:]).T+self.b[1]
                y[2]=np.outer(self.A[2][0],X[0,:]).T+\
                     np.outer(self.A[2][1],X[1,:]).T+\
                     np.outer(self.A[2][2],X[2,:]).T+self.b[2]
            else: # TODO check this could have error
                y[0]=np.outer(self.A[0][0][tind],X[0,:]).T+\
                     np.outer(self.A[0][1][tind],X[1,:]).T+\
                     np.outer(self.A[0][2][tind],X[2,:]).T+self.b[0][tind]
                y[1]=np.outer(self.A[1][0][tind],X[0,:]).T+\
                     np.outer(self.A[1][1][tind],X[1,:]).T+\
                     np.outer(self.A[1][2][tind],X[2,:]).T+self.b[1][tind]
                y[2]=np.outer(self.A[2][0][tind],X[0,:]).T+\
                     np.outer(self.A[2][1][tind],X[1,:]).T+\
                     np.outer(self.A[2][2][tind],X[2,:]).T+self.b[2][tind]
            y[0]=y[0].T
            y[1]=y[1].T
            y[2]=y[2].T
        else:
             raise NotImplementedError("MappingAffine.F: given dimension not implemented yet!")
        return y

    def invF(self,x,tind=None):
        """Inverse map F^{-1}(x)=A^{-1}(x-b)."""
        Y={}
        y={}
        if self.dim==2:
            if tind is None:
                Y[0]=x[0].T-self.b[0]
                Y[1]=x[1].T-self.b[1]
                y[0]=self.invA[0][0]*Y[0]+self.invA[0][1]*Y[1]
                y[1]=self.invA[1][0]*Y[0]+self.invA[1][1]*Y[1]
            else:
                Y[0]=x[0].T-self.b[0][tind]
                Y[1]=x[1].T-self.b[1][tind]
                y[0]=self.invA[0][0][tind]*Y[0]+self.invA[0][1][tind]*Y[1]
                y[1]=self.invA[1][0][tind]*Y[0]+self.invA[1][1][tind]*Y[1]
            y[0]=y[0].T
            y[1]=y[1].T
        elif self.dim==3:
            if tind is None:
                Y[0]=x[0].T-self.b[0]
                Y[1]=x[1].T-self.b[1]
                Y[2]=x[2].T-self.b[2]
                y[0]=self.invA[0][0]*Y[0]+self.invA[0][1]*Y[1]+self.invA[0][2]*Y[2]
                y[1]=self.invA[1][0]*Y[0]+self.invA[1][1]*Y[1]+self.invA[1][2]*Y[2]
                y[2]=self.invA[2][0]*Y[0]+self.invA[2][1]*Y[1]+self.invA[2][2]*Y[2]
            else:
                Y[0]=x[0].T-self.b[0][tind]
                Y[1]=x[1].T-self.b[1][tind]
                Y[2]=x[2].T-self.b[2][tind]
                y[0]=self.invA[0][0][tind]*Y[0]+self.invA[0][1][tind]*Y[1]+self.invA[0][2][tind]*Y[2]
                y[1]=self.invA[1][0][tind]*Y[0]+self.invA[1][1][tind]*Y[1]+self.invA[1][2][tind]*Y[2]
                y[2]=self.invA[2][0][tind]*Y[0]+self.invA[2][1][tind]*Y[1]+self.invA[2][2][tind]*Y[2]
            y[0]=y[0].T
            y[1]=y[1].T
            y[2]=y[2].T
        else:
             raise NotImplementedError("MappingAffine.F: given dimension not implemented yet!")
        return y

    def G(self,X,find=None):
        """Boundary mapping G(X)=Bx+c."""
        y={}
        if self.dim==2:
            if find is None:
                y[0]=np.outer(self.B[0],X).T+self.c[0]
                y[1]=np.outer(self.B[1],X).T+self.c[1]
            else:
                y[0]=np.outer(self.B[0][find],X).T+self.c[0][find]
                y[1]=np.outer(self.B[1][find],X).T+self.c[1][find]
            y[0]=y[0].T
            y[1]=y[1].T
        elif self.dim==3:
            if find is None:
                y[0]=np.outer(self.B[0][0],X[0,:]).T+np.outer(self.B[0][1],X[1,:]).T+self.c[0]
                y[1]=np.outer(self.B[1][0],X[0,:]).T+np.outer(self.B[1][1],X[1,:]).T+self.c[1]
                y[2]=np.outer(self.B[2][0],X[0,:]).T+np.outer(self.B[2][1],X[1,:]).T+self.c[2]
            else:
                y[0]=np.outer(self.B[0][0][find],X[0,:]).T+np.outer(self.B[0][1][find],X[1,:]).T+self.c[0][find]
                y[1]=np.outer(self.B[1][0][find],X[0,:]).T+np.outer(self.B[1][1][find],X[1,:]).T+self.c[1][find]
                y[2]=np.outer(self.B[2][0][find],X[0,:]).T+np.outer(self.B[2][1][find],X[1,:]).T+self.c[2][find]
            y[0]=y[0].T
            y[1]=y[1].T
            y[2]=y[2].T
        else:
            raise NotImplementedError("MappingAffine.G: given dimension not implemented yet!")
        return y

    def DF(self,X,tind=None):
        A=copy.deepcopy(self.A)
        
        if isinstance(X,dict):
            N=X[0].shape[1]
        else:
            N=X.shape[1]
        
        if self.dim==2:
            if tind is None: # TODO did not test
                A[0][0]=np.tile(A[0][0],(N,1)).T
                A[0][1]=np.tile(A[0][1],(N,1)).T
                A[1][0]=np.tile(A[1][0],(N,1)).T
                A[1][1]=np.tile(A[1][1],(N,1)).T
            else:
                A[0][0]=np.tile(A[0][0][tind],(N,1)).T
                A[0][1]=np.tile(A[0][1][tind],(N,1)).T
                A[1][0]=np.tile(A[1][0][tind],(N,1)).T
                A[1][1]=np.tile(A[1][1][tind],(N,1)).T
        if self.dim==3:
            if tind is None: # TODO did not test
                A[0][0]=np.tile(A[0][0],(N,1)).T
                A[0][1]=np.tile(A[0][1],(N,1)).T
                A[0][2]=np.tile(A[0][2],(N,1)).T
                A[1][0]=np.tile(A[1][0],(N,1)).T
                A[1][1]=np.tile(A[1][1],(N,1)).T
                A[1][2]=np.tile(A[1][2],(N,1)).T
                A[2][0]=np.tile(A[2][0],(N,1)).T
                A[2][1]=np.tile(A[2][1],(N,1)).T
                A[2][2]=np.tile(A[2][2],(N,1)).T
            else:
                A[0][0]=np.tile(A[0][0][tind],(N,1)).T
                A[0][1]=np.tile(A[0][1][tind],(N,1)).T
                A[0][2]=np.tile(A[0][2][tind],(N,1)).T
                A[1][0]=np.tile(A[1][0][tind],(N,1)).T
                A[1][1]=np.tile(A[1][1][tind],(N,1)).T
                A[1][2]=np.tile(A[1][2][tind],(N,1)).T
                A[2][0]=np.tile(A[2][0][tind],(N,1)).T
                A[2][1]=np.tile(A[2][1][tind],(N,1)).T
                A[2][2]=np.tile(A[2][2][tind],(N,1)).T           
                
        return A
        
    def detDF(self,X,tind=None):
        if tind is None:
            detDF=self.detA
        else:
            detDF=self.detA[tind]
        return np.tile(detDF,(X.shape[1],1)).T  
        
    def detDG(self,X,find=None):
        if find is None:
            detDG=self.detB
        else:
            detDG=self.detB[find]
        return np.tile(detDG,(X.shape[1],1)).T

    def normals(self,X,tind,find,t2f):
        N={}

        if self.dim==2:
            nref=np.array([[0.0,-1.0],[1.0,1.0],[-1.0,0.0]])
        elif self.dim==3:
            nref=np.array([[0.0,0.0,-1.0],[0.0,-1.0,0.0],[-1.0,0.0,0.0],[1.0,1.0,1.0]])
        else:
            raise NotImplementedError("MappingAffine.normals() not implemented for the used self.dim.")

        invDF=self.invDF(X,tind)

        # initialize n to zero
        n={}
        n[0]=np.zeros(find.shape[0]) # of size Nfacets x Nqp
        n[1]=np.zeros(find.shape[0])
        if self.dim==3:
            n[2]=np.zeros(find.shape[0])

        # compute all local normals
        for itr in range(nref.shape[0]):
            inds=np.nonzero(t2f[itr,tind]==find)[0]
            for jtr in range(nref.shape[1]):
               n[jtr][inds]=nref[itr,jtr]

        # map to global normals
        if self.dim==2:
            N[0]=invDF[0][0]*n[0][:,None]+invDF[1][0]*n[1][:,None]
            N[1]=invDF[0][1]*n[0][:,None]+invDF[1][1]*n[1][:,None]
        elif self.dim==3:
            N[0]=invDF[0][0]*n[0][:,None]+invDF[1][0]*n[1][:,None]+invDF[2][0]*n[2][:,None]
            N[1]=invDF[0][1]*n[0][:,None]+invDF[1][1]*n[1][:,None]+invDF[2][1]*n[2][:,None]
            N[2]=invDF[0][2]*n[0][:,None]+invDF[1][2]*n[1][:,None]+invDF[2][2]*n[2][:,None]
        else:
            raise NotImplementedError("MappingAffine.normals() not implemented for the used self.dim.")

        # normalize
        if self.dim==2:
            nlen=np.sqrt(N[0]**2+N[1]**2)
        elif self.dim==3:
            nlen=np.sqrt(N[0]**2+N[1]**2+N[2]**2)
        else:
            raise NotImplementedError("MappingAffine.normals() not implemented for the used self.dim.")

        # shrink to required facets and tile
        N[0]=N[0]/nlen
        N[1]=N[1]/nlen
        if self.dim==3:
            N[2]=N[2]/nlen

        return N # n[0] etc. are of size Nfacets x Nqp
        
    def invDF(self,X,tind=None):
        invA=copy.deepcopy(self.invA)
        
        if isinstance(X,dict):
            N=X[0].shape[1]
        else:
            N=X.shape[1]
        
        if self.dim==2:
            if tind is None: # TODO did not test
                invA[0][0]=np.tile(invA[0][0],(N,1)).T
                invA[0][1]=np.tile(invA[0][1],(N,1)).T
                invA[1][0]=np.tile(invA[1][0],(N,1)).T
                invA[1][1]=np.tile(invA[1][1],(N,1)).T
            else:
                invA[0][0]=np.tile(invA[0][0][tind],(N,1)).T
                invA[0][1]=np.tile(invA[0][1][tind],(N,1)).T
                invA[1][0]=np.tile(invA[1][0][tind],(N,1)).T
                invA[1][1]=np.tile(invA[1][1][tind],(N,1)).T
        elif self.dim==3:
            if tind is None: # TODO did not test
                invA[0][0]=np.tile(invA[0][0],(N,1)).T
                invA[0][1]=np.tile(invA[0][1],(N,1)).T
                invA[0][2]=np.tile(invA[0][2],(N,1)).T
                invA[1][0]=np.tile(invA[1][0],(N,1)).T
                invA[1][1]=np.tile(invA[1][1],(N,1)).T
                invA[1][2]=np.tile(invA[1][2],(N,1)).T
                invA[2][0]=np.tile(invA[2][0],(N,1)).T
                invA[2][1]=np.tile(invA[2][1],(N,1)).T
                invA[2][2]=np.tile(invA[2][2],(N,1)).T
            else:
                invA[0][0]=np.tile(invA[0][0][tind],(N,1)).T
                invA[0][1]=np.tile(invA[0][1][tind],(N,1)).T
                invA[0][2]=np.tile(invA[0][2][tind],(N,1)).T
                invA[1][0]=np.tile(invA[1][0][tind],(N,1)).T
                invA[1][1]=np.tile(invA[1][1][tind],(N,1)).T
                invA[1][2]=np.tile(invA[1][2][tind],(N,1)).T
                invA[2][0]=np.tile(invA[2][0][tind],(N,1)).T
                invA[2][1]=np.tile(invA[2][1][tind],(N,1)).T
                invA[2][2]=np.tile(invA[2][2][tind],(N,1)).T           
                
        return invA
