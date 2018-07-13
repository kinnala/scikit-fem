import unittest
import numpy as np
from skfem import *

"""
Check the convergence rates of the elements.
"""

class ConvergenceQ1(unittest.TestCase):
    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.1
    def runTest(self):
        @bilinear_form
        def laplace(u, du, v, dv, w):
            if du.shape[0] == 1:
                return du[0]*dv[0]
            elif du.shape[0] == 2:
                return du[0]*dv[0] + du[1]*dv[1]
            elif du.shape[0] == 3:
                return du[0]*dv[0] + du[1]*dv[1] + du[2]*dv[2]
            else:
                raise Exception("The dimension not supported")
        @linear_form
        def load(v, dv, w):
            x = w[0]
            if x.shape[0] == 1:
                return np.sin(np.pi*x[0])*(np.pi**2)*v
            elif x.shape[0] == 2:
                return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*(2.0*np.pi**2)*v
            elif x.shape[0] == 3:
                return np.sin(np.pi*x[0])*np.sin(np.pi*x[1])*np.sin(np.pi*x[2])*(3.0*np.pi**2)*v
            else:
                raise Exception("The dimension not supported")

        m = self.mesh
        Nitrs = 3
        L2s = np.zeros(Nitrs)
        H1s = np.zeros(Nitrs)
        hs = np.zeros(Nitrs)

        for itr in range(Nitrs):
            ib = self.create_basis(m)

            A = asm(laplace, ib)
            b = asm(load, ib)

            if m.dim() == 1: # TODO works only for elements with one DOF/node
                D = [0, 1]
                x = np.zeros(ib.dofnum.N)
            else:
                x, D = ib.find_dofs()
            I = ib.dofnum.complement_dofs(D)

            x[I] = solve(*condense(A, b, I=I))

            # calculate error
            L2s[itr], H1s[itr] = self.compute_error(m, ib, x)
            hs[itr] = m.param()

            m.refine()

        rateL2 = np.polyfit(np.log(hs), np.log(L2s), 1)[0]
        rateH1 = np.polyfit(np.log(hs), np.log(H1s), 1)[0]

        self.assertLess(np.abs(rateL2-self.rateL2), self.eps)
        self.assertLess(np.abs(rateH1-self.rateH1), self.eps)

    def compute_error(self, m, basis, U):
        uh, duh = basis.interpolate(U, derivative=True)
        dx = basis.dx
        x = basis.global_coordinates()

        def u(y):
            if y.shape[0] == 1:
                return np.sin(np.pi*y[0])
            elif y.shape[0] == 2:
                return np.sin(np.pi*y[0])*np.sin(np.pi*y[1])
            elif y.shape[0] == 3:
                return np.sin(np.pi*y[0])*np.sin(np.pi*y[1])*np.sin(np.pi*y[2])
            else:
                raise Exception("The dimension not supported")

        L2 = np.sqrt(np.sum(np.sum((uh - u(x))**2*dx, axis=1)))

        def ux(y):
            if y.shape[0] == 1:
                return np.pi*np.cos(np.pi*y[0])
            elif y.shape[0] == 2:
                return np.pi*np.cos(np.pi*y[0])*np.sin(np.pi*y[1])
            elif y.shape[0] == 3:
                return np.pi*np.cos(np.pi*y[0])*np.sin(np.pi*y[1])*np.sin(np.pi*y[2])
            else:
                raise Exception("The dimension not supported")

        if x.shape[0] >= 2:
            def uy(y):
                if y.shape[0] == 2:
                    return np.pi*np.sin(np.pi*y[0])*np.cos(np.pi*y[1])
                elif y.shape[0] == 3:
                    return np.pi*np.sin(np.pi*y[0])*np.cos(np.pi*y[1])*np.sin(np.pi*y[2])
                else:
                    raise Exception("The dimension not supported")

        if x.shape[0] == 3:
            def uz(y):
                return np.pi*np.sin(np.pi*y[0])*np.sin(np.pi*y[1])*np.cos(np.pi*y[2])


        if x.shape[0] == 3:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x))**2 +\
                                        (duh[1] - uy(x))**2 +\
                                        (duh[2] - uz(x))**2)*dx, axis=1)))
        elif x.shape[0] == 2:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x))**2 +\
                                        (duh[1] - uy(x))**2)*dx, axis=1)))
        else:
            H1 = np.sqrt(np.sum(np.sum(((duh[0] - ux(x))**2)*dx, axis=1)))

        return L2, H1


    def create_basis(self, m):
        e = ElementQ1()
        map = MappingIsoparametric(m, e)
        return InteriorBasis(m, e, map, 2)

    def setUp(self):
        self.mesh = MeshQuad()
        self.mesh.refine(2)

class ConvergenceQ2(ConvergenceQ1):
    """
    It seems that superconvergence 
    occurs here. Possibly due to
    the symmetricity of the loading
    and the presence of higher order
    symmetric basis functions?
    """
    rateL2 = 3.0
    rateH1 = 3.0
    def create_basis(self, m):
        e = ElementQ2()
        emap = ElementQ1()
        map = MappingIsoparametric(m, emap)
        return InteriorBasis(m, e, map, 3)
    def setUp(self):
        self.mesh = MeshQuad()
        self.mesh.refine(2)

class ConvergenceTriP1(ConvergenceQ1):
    def create_basis(self, m):
        e = ElementTriP1()
        map = MappingAffine(m)
        return InteriorBasis(m, e, map, 2)

    def setUp(self):
        self.mesh = MeshTri.init_sqsymmetric()
        self.mesh.refine(2)

class ConvergenceTriP2(ConvergenceTriP1):
    rateL2 = 3.0
    rateH1 = 2.0
    def create_basis(self, m):
        e = ElementTriP2()
        map = MappingAffine(m)
        return InteriorBasis(m, e, map, 2)

class ConvergenceHex1(ConvergenceQ1):
    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.11
    def create_basis(self, m):
        e = ElementHex1()
        map = MappingIsoparametric(m, e)
        return InteriorBasis(m, e, map, 3)
    def setUp(self):
        self.mesh = MeshHex()
        self.mesh.refine(2)

class ConvergenceTetP1(ConvergenceQ1):
    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.13
    def create_basis(self, m):
        e = ElementTetP1()
        map = MappingAffine(m)
        return InteriorBasis(m, e, map, 2)
    def setUp(self):
        self.mesh = MeshTet()
        self.mesh.refine(2)

class ConvergenceTetP2(ConvergenceTetP1):
    rateL2 = 3.0
    rateH1 = 2.0
    eps = 0.132
    def create_basis(self, m):
        e = ElementTetP2()
        map = MappingAffine(m)
        return InteriorBasis(m, e, map, 3)
    def setUp(self):
        self.mesh = MeshTet()
        self.mesh.refine(1)

class ConvergenceLineP1(ConvergenceQ1):
    def create_basis(self, m):
        e = ElementLineP1()
        return InteriorBasis(m, e)
    def setUp(self):
        self.mesh = MeshLine()
        self.mesh.refine(3)
