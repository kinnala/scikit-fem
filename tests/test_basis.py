from unittest import TestCase

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from skfem import BilinearForm, asm, solve, condense, projection
from skfem.mesh import MeshTri, MeshTet, MeshHex, MeshQuad, MeshLine
from skfem.assembly import CellBasis, FacetBasis, Dofs, Functional
from skfem.element import (ElementVectorH1, ElementTriP2, ElementTriP1,
                           ElementTetP2, ElementHexS2, ElementHex2,
                           ElementQuad2, ElementLineP2, ElementTriP0,
                           ElementLineP0, ElementQuad1, ElementQuad0,
                           ElementTetP1, ElementTetP0, ElementHex1,
                           ElementHex0, ElementLineP1, ElementLineMini)


class TestCompositeSplitting(TestCase):

    def runTest(self):
        """Solve Stokes problem, try splitting and other small things."""

        m = MeshTri().refined()
        m = m.refined(3).with_boundaries({
            'up': lambda x: x[1] == 1.,
            'rest': lambda x: x[1] != 1.,
        })

        e = ElementVectorH1(ElementTriP2()) * ElementTriP1()

        basis = CellBasis(m, e)

        @BilinearForm
        def bilinf(u, p, v, q, w):
            from skfem.helpers import grad, ddot, div
            return (ddot(grad(u), grad(v)) - div(u) * q - div(v) * p
                    - 1e-2 * p * q)

        S = asm(bilinf, basis)

        D = basis.find_dofs(skip=['u^2'])
        x = basis.zeros()
        x[D['up'].all('u^1^1')] = .1

        x = solve(*condense(S, x=x, D=D))

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

        m = self.mesh_type().refined(2)

        basis = CellBasis(m, self.elem_type())

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
        m = self.mesh_type().refined(self.nrefs)
        basis = CellBasis(m, self.element_type())
        x = projection(lambda x: x[0] ** 2, basis)
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


class TestInterpolatorHex(TestInterpolatorTet):

    mesh_type = MeshHex
    element_type = ElementHex2


class TestInterpolatorLine(TestInterpolatorTet):

    mesh_type = MeshLine
    element_type = ElementLineP2


class TestInterpolatorLine2(TestInterpolatorTet):

    mesh_type = MeshLine
    element_type = ElementLineP2
    nrefs = 5


class TestIncompatibleMeshElement(TestCase):

    def runTest(self):

        with self.assertRaises(ValueError):
            m = MeshTri()
            e = ElementTetP2()
            basis = CellBasis(m, e)


@pytest.mark.parametrize(
    "mtype,e,nrefs,npoints",
    [
        (MeshTri, ElementTriP1(), 0, 10),
        (MeshTri, ElementTriP1(), 1, 10),
        (MeshTri, ElementTriP1(), 5, 10),
        (MeshTri, ElementTriP1(), 1, 3e5),
        (MeshTet, ElementTetP1(), 1, 10),
        (MeshTet, ElementTetP1(), 5, 10),
        (MeshTet, ElementTetP1(), 1, 3e5),
        (MeshQuad, ElementQuad1(), 1, 10),
        (MeshQuad, ElementQuad1(), 1, 3e5),
    ]
)
def test_interpolator_probes(mtype, e, nrefs, npoints):

    m = mtype().refined(nrefs)

    np.random.seed(0)
    X = np.random.rand(m.p.shape[0], int(npoints))

    basis = CellBasis(m, e)

    y = projection(lambda x: x[0] ** 2, basis)

    assert_allclose(basis.probes(X) @ y, basis.interpolator(y)(X))
    assert_allclose(basis.probes(X) @ y, X[0] ** 2, atol=1e-1)


@pytest.mark.parametrize(
    "mtype,e1,e2",
    [
        (MeshTri, ElementTriP1(), ElementTriP0()),
        (MeshTri, ElementTriP1(), ElementTriP1()),
        (MeshTri, ElementTriP2(), ElementTriP1()),
        (MeshTri, ElementTriP2(), ElementTriP2()),
        (MeshTri, ElementTriP2(), None),
        (MeshQuad, ElementQuad1(), ElementQuad0()),
        (MeshQuad, ElementQuad1(), ElementQuad1()),
        (MeshQuad, ElementQuad2(), ElementQuad2()),
        (MeshTet, ElementTetP1(), ElementTetP0()),
        (MeshTet, ElementTetP2(), ElementTetP2()),
        (MeshHex, ElementHex1(), ElementHex0()),
        (MeshHex, ElementHex1(), ElementHex1()),
        (MeshHex, ElementHex2(), ElementHex2()),
    ]
)
def test_trace(mtype, e1, e2):

    m = mtype().refined(3)

    # use the boundary where last coordinate is zero
    basis = FacetBasis(m, e1,
                       facets=m.facets_satisfying(lambda x: x[x.shape[0] - 1] == 0.0))
    xfun = projection(lambda x: x[0], CellBasis(m, e1))
    nbasis, y = basis.trace(xfun, lambda p: p[0:(p.shape[0] - 1)], target_elem=e2)

    @Functional
    def integ(w):
        return w.y

    # integrate f(x) = x_1 over trace mesh
    assert_almost_equal(integ.assemble(nbasis, y=nbasis.interpolate(y)), .5)


@pytest.mark.parametrize(
    "etype",
    [ElementLineP1, ElementLineP2, ElementLineMini]
)
def test_point_source(etype):

    from skfem.models.poisson import laplace

    mesh = MeshLine().refined()
    basis = CellBasis(mesh, etype())
    source = np.array([0.7])
    u = solve(*condense(asm(laplace, basis), basis.point_source(source), D=basis.find_dofs()))
    exact = np.stack([(1 - source) * mesh.p, (1 - mesh.p) * source]).min(0)
    assert_almost_equal(u[basis.nodal_dofs], exact)
