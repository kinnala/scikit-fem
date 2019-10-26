"""Solve problems that have manufactured solutions."""
import unittest

import numpy as np

from skfem import *
from skfem.models.poisson import laplace, mass


class Line1D(unittest.TestCase):
    """Solve the following problem:

    u'' = 0
    u(0) = 0
    u'(1) = 1

    Solution is u(x) = x.

    """

    def runTest(self):
        m = MeshLine(np.linspace(0., 1.))
        m.refine(2)
        e = ElementLineP1()
        ib = InteriorBasis(m, e)
        fb = FacetBasis(m, e)

        @linear_form
        def boundary_flux(v, dv, w):
            return v * (w.x[0] == 1.)

        n = m.p.shape[-1]
        L = asm(laplace, ib)
        b = asm(boundary_flux, fb)
        D = m.nodes_satisfying(lambda x: x == 0.0)
        I = ib.complement_dofs(D)  # noqa E741
        u = solve(*condense(L, b, I=I))  # noqa E741

        self.assertTrue(np.sum(np.abs(u - m.p[0, :])) < 1e-10)


class LineNegative1D(unittest.TestCase):
    """Solve the following problem:

    u'' = 0
    u(0) = 0
    u'(1) = -1

    Solution is u(x) = -x.

    """

    def runTest(self):
        m = MeshLine(np.linspace(0., 1.))
        m.refine(2)
        e = ElementLineP1()
        ib = InteriorBasis(m, e)
        fb = FacetBasis(m, e)

        @linear_form
        def boundary_flux(v, dv, w):
            return -v * (w.x[0] == 1.)

        n = m.p.shape[-1]
        L = asm(laplace, ib)
        b = asm(boundary_flux, fb)
        D = m.nodes_satisfying(lambda x: x == 0.0)
        I = ib.complement_dofs(D)  # noqa E741
        u = solve(*condense(L, b, I=I))  # noqa E741

        self.assertTrue(np.sum(np.abs(u + m.p[0, :])) < 1e-10)


class LineNeumann1D(unittest.TestCase):
    """Solve the following problem:

    -u'' + eps*u = 0
    u'(0) = 1
    u'(1) = 1

    Solution is u(x) = x-0.5.

    """

    def runTest(self):
        m = MeshLine(np.linspace(0., 1.))
        m.refine(2)
        e = ElementLineP1()
        ib = InteriorBasis(m, e)
        fb = FacetBasis(m, e)

        @linear_form
        def boundary_flux(v, dv, w):
            return v * (w.x[0] == 1) - v * (w.x[0] == 0)

        n = m.p.shape[-1]
        L = asm(laplace, ib)
        M = asm(mass, ib)
        b = asm(boundary_flux, fb)
        u = solve(L + 1e-6 * M, b)

        self.assertTrue(np.sum(np.abs(u - m.p[0, :] + 0.5)) < 1e-4)


class TestExactHexElement(unittest.TestCase):
    mesh = MeshHex
    elem = ElementHex1
    funs = [
        lambda x: 1 + x[0] * x[1] * x[2],
        lambda x: 1 + x[0] * x[1] + x[1] * x[2] + x[0],
    ]

    def set_bc(self, fun, basis):
        return fun(basis.mesh.p)

    def runTest(self):
        @bilinear_form
        def dudv(u, du, v, dv, w):
            return sum(du * dv)

        m = self.mesh()
        m.refine(4)

        ib = InteriorBasis(m, self.elem())

        A = asm(dudv, ib)

        D = ib.get_dofs().all()
        I = ib.complement_dofs(D)

        for X in self.funs:
            x = self.set_bc(X, ib)
            Xh = x.copy()
            x = solve(*condense(A, 0 * x, x=x, I=I))
            self.assertLessEqual(np.sum(x - Xh), 1e-11)


class TestExactQuadElement(TestExactHexElement):
    mesh = MeshQuad
    elem = ElementQuad1
    funs = [
        lambda x: 1 + 0 * x[0],
        lambda x: 1 + x[0] + x[1] + x[0] * x[1],
    ]


class TestExactTetElement(TestExactHexElement):
    mesh = MeshTet
    elem = ElementTetP1
    funs = [
        lambda x: 1 + 0 * x[0],
        lambda x: 1 + x[0] + x[1],
    ]


class TestExactTriElementP2(TestExactHexElement):
    mesh = MeshTri
    elem = ElementTriP2
    funs = [
        lambda x: 1 + 0 * x[0],
        lambda x: 1 + x[0] + x[1] + x[0] * x[1],
    ]

    def set_bc(self, fun, basis):
        return fun(basis.doflocs)


class TestExactQuadElement2(TestExactTriElementP2):
    mesh = MeshQuad
    elem = ElementQuad2


if __name__ == '__main__':
    unittest.main()
