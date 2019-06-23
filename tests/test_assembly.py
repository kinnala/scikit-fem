"""Unit tests for assembly operations."""

import unittest
import numpy as np
from skfem import *


class IntegrateOneOverBoundaryQ1(unittest.TestCase):
    def createBasis(self):
        m = MeshQuad()
        m.refine(6)
        self.fbasis = FacetBasis(m, ElementQuad1())
        self.boundary_area = 4.0000

    def runTest(self):
        self.createBasis()

        @bilinear_form
        def uv(u, du, v, dv, w):
            return u * v

        B = asm(uv, self.fbasis)

        @linear_form
        def gv(v, dv, w):
            return 1.0 * v

        g = asm(gv, self.fbasis)

        ones = np.ones(g.shape)

        self.assertAlmostEqual(ones @ g, self.boundary_area, places=4)
        self.assertAlmostEqual(ones @ (B @ ones), self.boundary_area, places=4)


class IntegrateOneOverBoundaryHex1(IntegrateOneOverBoundaryQ1):
    def createBasis(self):
        m = MeshHex()
        m.refine(3)
        self.fbasis = FacetBasis(m, ElementHex1())
        self.boundary_area = 6.000


class IntegrateFuncOverBoundary(unittest.TestCase):
    def runTest(self):
        cases = [(MeshHex, ElementHex1),
                 (MeshTet, ElementTetP1),
                 (MeshTet, ElementTetP0)]

        for (mtype, etype) in cases:
            m = mtype()
            m.refine(3)
            fb = FacetBasis(m, etype())

            @bilinear_form
            def uv(u, du, v, dv, w):
                x, y, z = w.x
                return x**2*y**2*z**2*u*v

            B = asm(uv, fb)

            ones = np.ones(B.shape[0])

            self.assertAlmostEqual(ones @ (B @ ones), 0.3333333333, places=5)


class IntegrateFuncOverBoundaryPart(unittest.TestCase):
    case = (MeshHex, ElementHex1)
    
    def runTest(self):
        mtype, etype = self.case
        m = mtype()
        m.refine(3)
        bnd = m.facets_satisfying(lambda x: x[0]==1.0)
        fb = FacetBasis(m, etype(), facets=bnd)

        @bilinear_form
        def uv(u, du, v, dv, w):
            x, y, z = w.x
            return x**2*y**2*z**2*u*v

        B = asm(uv, fb)
        ones = np.ones(B.shape[0])

        self.assertAlmostEqual(ones @ (B @ ones), 0.11111111, places=5)


class IntegrateFuncOverBoundaryPartTetP1(IntegrateFuncOverBoundaryPart):
    case = (MeshTet, ElementTetP1)


class IntegrateFuncOverBoundaryPartTetP2(IntegrateFuncOverBoundaryPart):
    case = (MeshTet, ElementTetP2)


class IntegrateFuncOverBoundaryPartTetP0(IntegrateFuncOverBoundaryPart):
    case = (MeshTet, ElementTetP0)


class BasisInterpolator(unittest.TestCase):
    case = (MeshTri, ElementTriP1)

    def initOnes(self, basis):
        return np.ones(basis.N)

    def runTest(self):
        mtype, etype = self.case
        m = mtype()
        m.refine(3)
        e = etype()
        ib = InteriorBasis(m, e)

        x = self.initOnes(ib)
        f = ib.interpolator(x)

        self.assertTrue(np.sum(f(np.array([np.sin(m.p[0, :]),
                                           np.sin(3.*m.p[1, :])]))-1.) < 1e-10)


class BasisInterpolatorMorley(BasisInterpolator):
    case = (MeshTri, ElementTriMorley)

    def initOnes(self, basis):
        @bilinear_form
        def mass(u, du, ddu, v, dv, ddv, w):
            return u * v

        @linear_form
        def ones(v, dv, ddv, w):
            return 1.0*v

        M = asm(mass, basis)
        f = asm(ones, basis)

        return solve(M, f)


class NormalVectorTestTri(unittest.TestCase):
    case = (MeshTri(), ElementTriP1())

    def runTest(self):
        basis = FacetBasis(*self.case)
        @linear_form
        def linf(v, dv, w):
            return np.sum(w.n**2, axis=0)*v
        b = asm(linf, basis)
        self.assertAlmostEqual(b @ np.ones(b.shape),
                               2*self.case[0].p.shape[0],
                               places=10)


class NormalVectorTestTet(NormalVectorTestTri):
    case = (MeshTet(), ElementTetP1())


class NormalVectorTestTetP2(NormalVectorTestTri):
    case = (MeshTet(), ElementTetP2())


class NormalVectorTestQuad(NormalVectorTestTri):
    case = (MeshQuad(), ElementQuad1())


class NormalVectorTestHex(NormalVectorTestTri):
    case = (MeshHex(), ElementHex1())


if __name__ == '__main__':
    unittest.main()
