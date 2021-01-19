"""Solve problems that have manufactured solutions."""

import unittest
from pathlib import Path

import numpy as np

from skfem.models.poisson import laplace, mass, unit_load
from skfem.mesh import (MeshHex, MeshLine, MeshQuad, MeshTet,
                        MeshTri, MeshTri2, MeshQuad2)
from skfem.element import (ElementHex1, ElementHexS2,
                           ElementLineP1, ElementLineP2, ElementLineMini, 
                           ElementQuad1, ElementQuad2, ElementTetP1,
                           ElementTriP2, ElementHex2, ElementTriP1)
from skfem.assembly import FacetBasis, InteriorBasis
from skfem import asm, condense, solve, LinearForm


class Line1D(unittest.TestCase):
    """Solve the following problem:

    u'' = 0
    u(0) = 0
    u'(1) = 1

    Solution is u(x) = x.

    """

    e = ElementLineP1()

    def runTest(self):
        m = MeshLine(np.linspace(0., 1.)).refined(2)
        ib = InteriorBasis(m, self.e)
        fb = FacetBasis(m, self.e)

        @LinearForm
        def boundary_flux(v, w):
            return v * (w.x[0] == 1.)

        L = asm(laplace, ib)
        b = asm(boundary_flux, fb)
        D = m.nodes_satisfying(lambda x: x == 0.0)
        I = ib.complement_dofs(D)  # noqa E741
        u = solve(*condense(L, b, I=I))  # noqa E741

        np.testing.assert_array_almost_equal(u[ib.nodal_dofs[0]], m.p[0], -10)

class Line1DP2(Line1D):
    e = ElementLineP2()


class Line1DMini(Line1D):
    e = ElementLineMini()


class LineNegative1D(unittest.TestCase):
    """Solve the following problem:

    u'' = 0
    u(0) = 0
    u'(1) = -1

    Solution is u(x) = -x.

    """
    e = ElementLineP1()

    def runTest(self):
        m = MeshLine(np.linspace(0., 1.)).refined(2)
        ib = InteriorBasis(m, self.e)
        m.define_boundary('left' ,lambda x: x[0] == 0.0)
        m.define_boundary('right', lambda x: x[0] == 1.0)
        fb = FacetBasis(m, self.e, facets=m.boundaries['right'])

        @LinearForm
        def boundary_flux(v, w):
            return -w.x[0] * v

        L = asm(laplace, ib)
        b = asm(boundary_flux, fb)
        D = ib.find_dofs()['left'].all()
        I = ib.complement_dofs(D)  # noqa E741
        u = solve(*condense(L, b, I=I))  # noqa E741

        np.testing.assert_array_almost_equal(u[ib.nodal_dofs[0]], -m.p[0], -10)


class LineNegative1DP2(LineNegative1D):
    e = ElementLineP2()


class LineNegative1DMini(LineNegative1D):
    e = ElementLineMini()


class LineNeumann1D(unittest.TestCase):
    """Solve the following problem:

    -u'' + eps*u = 0
    u'(0) = 1
    u'(1) = 1

    Solution is u(x) = x-0.5.

    """
    e = ElementLineP1()

    def runTest(self):
        m = MeshLine(np.linspace(0., 1.)).refined(2)
        ib = InteriorBasis(m, self.e)
        fb = FacetBasis(m, self.e)

        @LinearForm
        def boundary_flux(v, w):
            return v * (w.x[0] == 1) - v * (w.x[0] == 0)

        L = asm(laplace, ib)
        M = asm(mass, ib)
        b = asm(boundary_flux, fb)
        u = solve(L + 1e-6 * M, b)

        np.testing.assert_array_almost_equal(u[ib.nodal_dofs[0]], m.p[0] - .5, -4)


class LineNeumann1DP2(LineNeumann1D):
    e = ElementLineP2()


class LineNeumann1DMini(LineNeumann1D):
    e = ElementLineMini()
    

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

        m = self.mesh().refined(3)

        ib = InteriorBasis(m, self.elem())

        A = asm(laplace, ib)

        D = ib.get_dofs().all()
        I = ib.complement_dofs(D)

        for X in self.funs:
            x = self.set_bc(X, ib)
            Xh = x.copy()
            x = solve(*condense(A, 0 * x, x=x, I=I))
            self.assertLessEqual(np.sum(x - Xh), 1e-10)


class TestExactHexS2(TestExactHexElement):

    elem = ElementHexS2
    funs = [
        lambda x: 1 + 0 * x[0],
    ]

    def set_bc(self, fun, basis):
        return fun(basis.doflocs)


class TestExactHex2(TestExactHexElement):

    elem = ElementHex2
    funs = [
        lambda x: 1 + 0 * x[0],
        lambda x: 1 + x[0] * x[1] * x[2],
    ]

    def set_bc(self, fun, basis):
        return fun(basis.doflocs)


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
        lambda x: 1 + x[0] + x[1] + x[2],
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


class SolveCirclePoisson(unittest.TestCase):

    mesh_type = MeshTri2
    element_type = ElementTriP1
    filename = "quadratic_tri.msh"

    def runTest(self):
        path = Path(__file__).parents[1] / 'docs' / 'examples' / 'meshes'
        m = self.mesh_type.load(path / self.filename)
        basis = InteriorBasis(m, self.element_type())

        A = laplace.assemble(basis)
        b = unit_load.assemble(basis)
        x = solve(*condense(A, b, D=basis.get_dofs()))

        self.assertAlmostEqual(np.max(x), 0.06261690318912218, places=3)


class SolveCirclePoissonQuad(SolveCirclePoisson):

    mesh_type = MeshQuad2
    element_type = ElementQuad1
    filename = "quadratic_quad.msh"


class SolveCirclePoissonQuad2(SolveCirclePoisson):

    mesh_type = MeshQuad2
    element_type = ElementQuad2
    filename = "quadratic_quad.msh"


class SolveCirclePoissonTri2(SolveCirclePoisson):

    mesh_type = MeshTri2
    element_type = ElementTriP2
    filename = "quadratic_tri.msh"



if __name__ == '__main__':
    unittest.main()
