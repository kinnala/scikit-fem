"""Solve problems that have manufactured solutions."""

from skfem.utils import penalize
from unittest import TestCase
from pathlib import Path

import pytest

import numpy as np
from skfem import (LinearForm, Functional, asm, condense, solve, projection,
                   enforce)
from skfem.assembly import FacetBasis, InteriorBasis
from skfem.element import (ElementHex1, ElementHex2, ElementHexS2,
                           ElementLineMini, ElementLineP1, ElementLineP2,
                           ElementQuad1, ElementQuad2, ElementTetP1,
                           ElementTetP2, ElementTriP1, ElementTriP2)
from skfem.mesh import (MeshHex, MeshLine, MeshQuad, MeshQuad2, MeshTet,
                        MeshTet2, MeshTri, MeshTri2)
from skfem.models.poisson import laplace, mass, unit_load
from skfem.helpers import dot


class Line1D(TestCase):
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


class LineNegative1D(TestCase):
    """Solve the following problem:

    u'' = 0
    u(0) = 0
    u'(1) = -1

    Solution is u(x) = -x.

    """
    e = ElementLineP1()

    def runTest(self):
        m = MeshLine(np.linspace(0., 1.)).refined(2).with_boundaries({
            'left': lambda x: x[0] == 0.0,
            'right': lambda x: x[0] == 1.0,
        })
        ib = InteriorBasis(m, self.e)
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


class LineNeumann1D(TestCase):
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
    

class TestExactHexElement(TestCase):

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
            x = solve(*condense(A, x=x, I=I))
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


class SolveCirclePoisson(TestCase):

    mesh_type = MeshTri2
    element_type = ElementTriP1
    filename = "quadratic_tri.msh"
    maxval = 0.06243516822727334

    def init_mesh(self):
        path = Path(__file__).parents[1] / 'docs' / 'examples' / 'meshes'
        return self.mesh_type.load(path / self.filename)

    def runTest(self):
        m = self.init_mesh()
        basis = InteriorBasis(m, self.element_type())

        A = laplace.assemble(basis)
        b = unit_load.assemble(basis)
        x = solve(*condense(A, b, D=basis.get_dofs()))

        self.assertAlmostEqual(np.max(x), self.maxval, places=3)


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


class SolveCirclePoissonTri2Init(SolveCirclePoissonTri2):

    def init_mesh(self):
        return self.mesh_type.init_circle().scaled(0.5)


class SolveCirclePoissonTet(SolveCirclePoisson):

    mesh_type = MeshTet
    element_type = ElementTetP1
    maxval = 0.0405901240018571

    def init_mesh(self):
        return self.mesh_type.init_ball().scaled(0.5)


class SolveCirclePoissonTet2(SolveCirclePoisson):

    mesh_type = MeshTet2
    element_type = ElementTetP2
    filename = "quadratic_sphere_tet.msh"
    maxval = 0.0405901240018571


@pytest.mark.parametrize(
    "mesh_elem", [(MeshTri, ElementTriP2()), (MeshQuad, ElementQuad2())]
)
@pytest.mark.parametrize("impose", [enforce, penalize])
def test_solving_inhomogeneous_laplace(mesh_elem, impose):
    """Adapted from example 14."""

    mesh, elem = mesh_elem

    m = mesh().refined(4)
    basis = InteriorBasis(m, elem)
    boundary_basis = FacetBasis(m, elem)
    boundary_dofs = boundary_basis.get_dofs().flatten()

    def dirichlet(x):
        """return a harmonic function"""
        return ((x[0] + 1.j * x[1]) ** 2).real

    u = basis.zeros()
    A = laplace.assemble(basis)
    u[boundary_dofs] = projection(dirichlet,
                                  boundary_basis,
                                  I=boundary_dofs)
    u = solve(*impose(A, x=u, D=boundary_dofs))


    @Functional
    def gradu(w):
        gradu = w['sol'].grad
        return dot(gradu, gradu)

    np.testing.assert_almost_equal(
        gradu.assemble(basis, sol=basis.interpolate(u)),
        8 / 3,
        decimal=9
    )


if __name__ == "__main__":
    import pytest
    import unittest

    unittest.main()
    pytest.main()    
