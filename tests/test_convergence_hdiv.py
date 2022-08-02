import unittest

import numpy as np

from skfem import BilinearForm, CellBasis, LinearForm, asm, solve
from skfem.element import (ElementTetP0, ElementTetRT0, ElementTriP0,
                           ElementTriRT0, ElementTriBDM1, ElementDG,
                           ElementQuadRT0, ElementQuad0, ElementTriRT2,
                           ElementTriP1, ElementHexRT1, ElementHex0)
from skfem.mesh import MeshTet, MeshTri, MeshQuad, MeshHex


class ConvergenceRaviartThomas(unittest.TestCase):
    rateL2 = 1.0
    rateHdiv = 1.0
    eps = 0.1
    Hdivbound = 0.04
    L2bound = 0.01

    def runTest(self):

        @BilinearForm
        def bilinf_A(sigma, tau, w):
            from skfem.helpers import dot
            return dot(sigma, tau)

        @BilinearForm
        def bilinf_B(sigma, v, w):
            return sigma.div * v

        @LinearForm
        def load(v, w):
            x = w.x
            if x.shape[0] == 1:
                return (np.sin(np.pi * x[0]) * (np.pi ** 2) * v)
            elif x.shape[0] == 2:
                return (np.sin(np.pi * x[0]) *
                        np.sin(np.pi * x[1]) * (2.0 * np.pi ** 2) * v)
            elif x.shape[0] == 3:
                return (np.sin(np.pi * x[0]) *
                        np.sin(np.pi * x[1]) *
                        np.sin(np.pi * x[2]) * (3.0 * np.pi ** 2) * v)
            else:
                raise Exception("The dimension not supported")

        m = self.mesh
        Nitrs = 3
        L2s = np.zeros(Nitrs)
        Hdivs = np.zeros(Nitrs)
        hs = np.zeros(Nitrs)

        for itr in range(Nitrs):
            ib1, ib2 = self.create_basis(m)

            A = asm(bilinf_A, ib1)
            B = asm(bilinf_B, ib1, ib2)
            b = np.concatenate((
                ib1.zeros(),
                -asm(load, ib2)
            ))

            from scipy.sparse import bmat
            K = bmat([[A, B.T], [B, None]]).tocsr()

            x = solve(K, b)

            # calculate error
            sigma, u = np.split(x, [A.shape[0]])

            L2s[itr] = self.compute_L2(m, ib2, u)
            Hdivs[itr] = self.compute_Hdiv(m, ib1, sigma)
            hs[itr] = m.param()

            m = m.refined()

        rateL2 = np.polyfit(np.log(hs), np.log(L2s), 1)[0]
        rateHdiv = np.polyfit(np.log(hs), np.log(Hdivs), 1)[0]

        self.assertLess(np.abs(rateL2 - self.rateL2),
                        self.eps,
                        msg='observed L2 rate: {}'.format(rateL2))
        self.assertLess(np.abs(rateHdiv - self.rateHdiv),
                        self.eps,
                        msg='observed Hdiv rate: {}'.format(rateHdiv))
        self.assertLess(Hdivs[-1], self.Hdivbound)
        self.assertLess(L2s[-1], self.L2bound)

    def compute_L2(self, m, basis, U):
        uh, *_ = basis.interpolate(U).astuple
        dx = basis.dx
        x = basis.global_coordinates()

        def u(y):
            if y.shape[0] == 1:
                return np.sin(np.pi * y[0])
            elif y.shape[0] == 2:
                return (np.sin(np.pi * y[0]) *
                        np.sin(np.pi * y[1]))
            return (np.sin(np.pi * y[0]) *
                    np.sin(np.pi * y[1]) *
                    np.sin(np.pi * y[2]))

        return np.sqrt(np.sum(np.sum((uh - u(x.value)) ** 2 * dx, axis=1)))

    def compute_Hdiv(self, m, basis, U):
        uh, duh, *_ = basis.interpolate(U).astuple
        dx = basis.dx
        x = basis.global_coordinates()

        def divu(y):
            if y.shape[0] == 1:
                return np.pi * np.cos(np.pi * y[0])
            elif y.shape[0] == 2:
                return (np.pi * np.cos(np.pi * y[0]) * np.sin(np.pi * y[1]) +
                        np.pi * np.sin(np.pi * y[0]) * np.cos(np.pi * y[1]))
            return np.pi * (np.cos(np.pi * y[0]) *
                            np.sin(np.pi * y[1]) *
                            np.sin(np.pi * y[2]) +
                            np.sin(np.pi * y[0]) *
                            np.cos(np.pi * y[1]) *
                            np.sin(np.pi * y[2]) +
                            np.sin(np.pi * y[0]) *
                            np.sin(np.pi * y[1]) *
                            np.cos(np.pi * y[2]))

        divuh = sum(uh)

        return np.sqrt(np.sum(np.sum(((divuh - divu(x.value)) ** 2) * dx, axis=1)))

    def create_basis(self, m):
        e = ElementTriRT0()
        e0 = ElementTriP0()
        return (CellBasis(m, e, intorder=2),
                CellBasis(m, e0, intorder=2))

    def setUp(self):
        self.mesh = MeshTri().refined(4)


class ConvergenceBDM1(ConvergenceRaviartThomas):
    rateL2 = 1.
    rateHdiv = 2.
    eps = 0.1
    Hdivbound = 0.05
    L2bound = 0.05

    def create_basis(self, m):
        e = ElementTriBDM1()
        e0 = ElementTriP0()
        return (CellBasis(m, e, intorder=4),
                CellBasis(m, e0, intorder=4))

    def setUp(self):
        self.mesh = MeshTri().refined(3)


class ConvergenceRT2(ConvergenceRaviartThomas):
    rateL2 = 2.
    rateHdiv = 2.
    eps = 0.1
    Hdivbound = 0.05
    L2bound = 0.05

    def create_basis(self, m):
        e = ElementTriRT2()
        e0 = ElementDG(ElementTriP1())
        return (CellBasis(m, e, intorder=4),
                CellBasis(m, e0, intorder=4))

    def setUp(self):
        self.mesh = MeshTri().refined(3)


class ConvergenceQuadRT0(ConvergenceRaviartThomas):
    Hdivbound = 0.07
    L2bound = 0.03

    def create_basis(self, m):
        b = CellBasis(m, ElementQuadRT0())
        return (b, b.with_element(ElementQuad0()))

    def setUp(self):
        self.mesh = MeshQuad().refined(3)


class ConvergenceRaviartThomas3D(ConvergenceRaviartThomas):
    rateL2 = 0.5
    rateHdiv = 1.0
    eps = 0.1
    Hdivbound = 0.3
    L2bound = 0.05

    def create_basis(self, m):
        e = ElementTetRT0()
        e0 = ElementTetP0()
        return (CellBasis(m, e, intorder=2),
                CellBasis(m, e0, intorder=2))

    def setUp(self):
        self.mesh = MeshTet().refined(1)


class ConvergenceRaviartThomas3DHex(ConvergenceRaviartThomas3D):

    rateL2 = 1.0
    L2bound = 0.1
    rateHdiv = 1.0
    Hdivbound = 0.31

    def create_basis(self, m):
        e = ElementHexRT1()
        e0 = ElementHex0()
        return (CellBasis(m, e, intorder=2),
                CellBasis(m, e0, intorder=2))

    def setUp(self):
        self.mesh = MeshHex().refined(1)
