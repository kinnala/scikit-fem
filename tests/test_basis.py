from unittest import TestCase

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from skfem import BilinearForm, asm, solve, condense, project
from skfem.mesh import MeshTri, MeshTet, MeshHex, MeshQuad, MeshLine
from skfem.assembly import InteriorBasis, FacetBasis, Dofs, Functional
from skfem.element import (ElementVectorH1, ElementTriP2, ElementTriP1,
                           ElementTetP2, ElementHexS2, ElementHex2,
                           ElementQuad2, ElementLineP2, ElementTriP0,
                           ElementLineP0, ElementQuad1, ElementQuad0,
                           ElementTetP1, ElementTetP0, ElementHex1,
                           ElementHex0, ElementLineP1)


class TestCompositeSplitting(TestCase):

    def runTest(self):
        """Solve Stokes problem, try splitting and other small things."""

        m = MeshTri()
        m.refine()
        m.define_boundary('centreline', lambda x: x[0] == .5,
                          boundaries_only=False)
        m.refine(3)

        e = ElementVectorH1(ElementTriP2()) * ElementTriP1()

        m.define_boundary('up', lambda x: x[1] == 1.)
        m.define_boundary('rest', lambda x: x[1] != 1.)

        basis = InteriorBasis(m, e)
        self.assertEqual(
            basis.get_dofs(m.boundaries['centreline']).all().size,
            (2 + 1) * (2**(1 + 3) + 1) + 2 * 2**(1 + 3))
        self.assertEqual(basis.find_dofs()['centreline'].all().size,
                         (2 + 1) * (2**(1 + 3) + 1) + 2 * 2**(1 + 3))

        @BilinearForm
        def bilinf(u, p, v, q, w):
            from skfem.helpers import grad, ddot, div
            return (ddot(grad(u), grad(v)) - div(u) * q - div(v) * p
                    - 1e-2 * p * q)

        S = asm(bilinf, basis)

        D = basis.find_dofs(skip=['u^2'])
        x = basis.zeros()
        x[D['up'].all('u^1^1')] = .1

        x = solve(*condense(S, basis.zeros(), x=x, D=D))

        (u, u_basis), (p, p_basis) = basis.split(x)

        self.assertEqual(len(u), m.p.shape[1] * 2 + m.facets.shape[1] * 2)
        self.assertEqual(len(p), m.p.shape[1])

        self.assertTrue(np.sum(p - x[basis.nodal_dofs[2]]) < 1e-8)

        U, P = basis.interpolate(x)
        self.assertTrue(isinstance(U.value, np.ndarray))
        self.assertTrue(isinstance(P.value, np.ndarray))

        self.assertTrue((basis.doflocs[:, D['up'].all()][1] == 1.).all())


class TestCompositeFacetAssembly(TestCase):

    def runTest(self):

        m = MeshTri()

        fbasis1 = FacetBasis(m, ElementTriP1() * ElementTriP1(),
                             facets=m.facets_satisfying(lambda x: x[0] == 0))
        fbasis2 = FacetBasis(m, ElementTriP1(),
                             facets=m.facets_satisfying(lambda x: x[0] == 0))

        @BilinearForm
        def uv1(u, p, v, q, w):
            return u * v + p * q

        @BilinearForm
        def uv2(u, v, w):
            return u * v

        A = asm(uv1, fbasis1)
        B = asm(uv2, fbasis2)

        assert_allclose(A[0].todense()[0, ::2],
                        B[0].todense()[0])


class TestFacetExpansion(TestCase):

    mesh_type = MeshTet
    elem_type = ElementTetP2

    def runTest(self):

        m = self.mesh_type()
        m.refine(2)

        basis = InteriorBasis(m, self.elem_type())

        for fun in [lambda x: x[0] == 0,
                    lambda x: x[0] == 1,
                    lambda x: x[1] == 0,
                    lambda x: x[1] == 1,
                    lambda x: x[2] == 0,
                    lambda x: x[2] == 1]:
            arr1 = basis.find_dofs({
                'kek': m.facets_satisfying(fun)
            })['kek'].edge['u']
            arr2 = basis.edge_dofs[:, m.edges_satisfying(fun)]

            assert_allclose(arr1, arr2.flatten())


class TestFacetExpansionHexS2(TestFacetExpansion):

    mesh_type = MeshHex
    elem_type = ElementHexS2


class TestFacetExpansionHex2(TestFacetExpansionHexS2):

    elem_type = ElementHex2


class TestInterpolatorTet(TestCase):

    mesh_type = MeshTet
    element_type = ElementTetP2
    nrefs = 1

    def runTest(self):
        m = self.mesh_type()
        m.refine(self.nrefs)
        basis = InteriorBasis(m, self.element_type())
        x = project(lambda x: x[0] ** 2, basis_to=basis)
        fun = basis.interpolator(x)
        X = np.linspace(0, 1, 10)
        dim = m.dim()
        if dim == 3:
            y = fun(np.array([X, [0.31] * 10, [0.62] * 10]))
        elif dim == 2:
            y = fun(np.array([X, [0.31] * 10]))
        else:
            y = fun(np.array([X]))
        assert_allclose(y, X ** 2, atol=1e-10)


class TestInterpolatorTet2(TestInterpolatorTet):

    mesh_type = MeshTet
    element_type = ElementTetP2
    nrefs = 3


class TestInterpolatorTri(TestInterpolatorTet):

    mesh_type = MeshTri
    element_type = ElementTriP2


class TestInterpolatorQuad(TestInterpolatorTet):

    mesh_type = MeshQuad
    element_type = ElementQuad2


class TestInterpolatorLine(TestInterpolatorTet):

    mesh_type = MeshLine
    element_type = ElementLineP2


class TestInterpolatorLine2(TestInterpolatorTet):

    mesh_type = MeshLine
    element_type = ElementLineP2
    nrefs = 5


@pytest.mark.parametrize(
    "mtype,e1,e2",
    [
        (MeshTri, ElementTriP1(), ElementLineP0()),
        (MeshTri, ElementTriP1(), ElementLineP1()),
        (MeshTri, ElementTriP2(), ElementLineP1()),
        (MeshTri, ElementTriP2(), ElementLineP2()),
        (MeshTri, ElementTriP2(), None),
        (MeshQuad, ElementQuad1(), ElementLineP0()),
        (MeshQuad, ElementQuad1(), ElementLineP1()),
        (MeshQuad, ElementQuad2(), ElementLineP2()),
        (MeshTet, ElementTetP1(), ElementTriP0()),
        (MeshTet, ElementTetP2(), ElementTriP2()),
        (MeshHex, ElementHex1(), ElementQuad0()),
        (MeshHex, ElementHex1(), ElementQuad1()),
        (MeshHex, ElementHex2(), ElementQuad2()),
    ]
)
def test_trace(mtype, e1, e2):

    m = mtype()
    m.refine(3)

    # use the boundary where last coordinate is zero
    basis = FacetBasis(m, e1,
                       facets=m.facets_satisfying(lambda x: x[x.shape[0] - 1] == 0.0))
    xfun = project(lambda x: x[0], basis_to=InteriorBasis(m, e1))
    nbasis, y = basis.trace(xfun, lambda p: p[0:(p.shape[0] - 1)], target_elem=e2)

    @Functional
    def integ(w):
        return w.y

    # integrate f(x) = x_1 over trace mesh
    assert_almost_equal(integ.assemble(nbasis, y=nbasis.interpolate(y)), .5)
