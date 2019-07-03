# -*- coding: utf-8 -*- 
"""Check the convergence rates of the elements."""

import unittest
import numpy as np
from skfem import *
from skfem.models.poisson import laplace


class ConvergenceQ1(unittest.TestCase):
    rateL2 = 2.0
    rateH1 = 1.0
    eps = 0.1
    def runTest(self):
        
        @linear_form
        def load(v, dv, w):
            x = w.x
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
                I = m.interior_nodes()
                x = np.zeros(ib.N)
            else:
                D = ib.get_dofs().all()
                x = np.zeros(ib.N)
                I = ib.complement_dofs(D)

            x[I] = solve(*condense(A, b, I=I))

            # calculate error
            L2s[itr], H1s[itr] = self.compute_error(m, ib, x)
            hs[itr] = m.param()

            m.refine()

        rateL2 = np.polyfit(np.log(hs), np.log(L2s), 1)[0]
        rateH1 = np.polyfit(np.log(hs), np.log(H1s), 1)[0]

        self.assertLess(np.abs(rateL2-self.rateL2),
                        self.eps,
                        msg = 'observed L2 rate: {}'.format(rateL2))
        self.assertLess(np.abs(rateH1-self.rateH1),
                        self.eps,
                        msg = 'observed H1 rate: {}'.format(rateH1))

    def compute_error(self, m, basis, U):
        uh, duh = basis.interpolate(U)
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
        e = ElementQuad1()
        map = MappingIsoparametric(m, e)
        return InteriorBasis(m, e, map, 2)

    def setUp(self):
        self.mesh = MeshQuad()
        self.mesh.refine(2)


class ConvergenceQ2(ConvergenceQ1):
    """It seems that superconvergence occurs here. Possibly due to the
    symmetricity of the loading and the presence of higher order symmetric
    basis functions?

    """
    rateL2 = 3.0
    rateH1 = 3.0
    def create_basis(self, m):
        e = ElementQuad2()
        emap = ElementQuad1()
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


class TetP2Test(unittest.TestCase):
    """Test second order tetrahedral element and facet
    assembly."""
    case = (MeshTet, ElementTetP2)
    limits = (1.9, 2.2)
    preref = 1

    def runTest(self):
        @bilinear_form
        def dudv(u, du, v, dv, w):
            return sum(du * dv)

        @bilinear_form
        def uv(u, du, v, dv, w):
            return u * v

        def F(x, y, z):
            return 2*x**2 + 2*y**2 - 6*x*y*z

        @linear_form
        def fv(v, dv, w):
            return F(*w.x)*v

        def G(x, y, z):
            return (x==1)*(3-3*y**2+2*y*z**3)+\
                   (x==0)*(-y*z**3)+\
                   (y==1)*(1+x-3*x**2+2*x*z**3)+\
                   (y==0)*(1+x-x*z**3)+\
                   (z==1)*(1+x+4*x*y-x**2*y**2)+\
                   (z==0)*(1+x-x**2*y**2)

        @linear_form
        def gv(v, dv, w):
            return G(*w.x)*v

        hs = np.array([])
        H1err = np.array([])
        L2err = np.array([])

        for itr in range(0,3):
            m = self.case[0]()
            m.refine(self.preref)
            m.refine(itr)

            ib = InteriorBasis(m, self.case[1]())
            fb = FacetBasis(m, self.case[1]())

            A = asm(dudv, ib)
            f = asm(fv, ib)

            B = asm(uv, fb)
            g = asm(gv, fb)

            u = np.zeros(ib.N)

            u = solve(A + B, f + g)

            L2, H1 = self.compute_error(m, ib, u)
            hs = np.append(hs, m.param())
            L2err = np.append(L2err, L2)
            H1err = np.append(H1err, H1)

        print(L2err)
        print(H1err)

        pfit = np.polyfit(np.log10(hs),
                          np.log10(np.sqrt(L2err**2+H1err**2)), 1)
        self.assertGreater(pfit[0], self.limits[0])
        self.assertLess(pfit[0], self.limits[1])

    def compute_error(self, m, basis, U):
        uh, duh = basis.interpolate(U)
        dx = basis.dx
        x = basis.global_coordinates()

        def u(x):
            return 1+x[0]-x[0]**2*x[1]**2+x[0]*x[1]*x[2]**3

        def ux(x):
            return 1-2*x[0]*x[1]**2+x[1]*x[2]**3

        def uy(x):
            return -2*x[0]**2*x[1]+x[0]*x[2]**3

        def uz(x):
            return 3*x[0]*x[1]*x[2]**2

        L2 = np.sqrt(np.sum((uh - u(x))**2*dx))

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


class TestHexFacet(TetP2Test):
    case = (MeshHex, ElementHex1)
    limits = (0.9, 1.1)
    preref = 2


if __name__ == '__main__':
    unittest.main()
