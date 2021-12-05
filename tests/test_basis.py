import pickle
from unittest import TestCase

import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal

from skfem import BilinearForm, LinearForm, asm, solve, condense, projection
from skfem.mesh import (MeshTri, MeshTet, MeshHex,
                        MeshQuad, MeshLine1, MeshWedge1)
from skfem.assembly import (CellBasis, FacetBasis, Dofs, Functional,
                            MortarFacetBasis)
from skfem.mapping import MappingIsoparametric, MappingMortar
from skfem.element import (ElementVectorH1, ElementTriP2, ElementTriP1,
                           ElementTetP2, ElementHexS2, ElementHex2,
                           ElementQuad2, ElementLineP2, ElementTriP0,
                           ElementLineP0, ElementQuad1, ElementQuad0,
                           ElementTetP1, ElementTetP0, ElementHex1,
                           ElementHex0, ElementLineP1, ElementLineMini,
                           ElementWedge1)
from skfem.models.poisson import laplace


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

        # test blocks splitting of forms while at it
        C1 = asm(bilinf.block(1, 1), CellBasis(m, ElementTriP1()))
        C2 = S[basis.nodal_dofs[-1]].T[basis.nodal_dofs[-1]].T
        self.assertTrue(abs((C1 - C2).min()) < 1e-10)
        self.assertTrue(abs((C1 - C2).max()) < 1e-10)

        # test splitting ElementVector
        (ux, uxbasis), (uy, uybasis) = u_basis.split(u)
        assert_allclose(ux[uxbasis.nodal_dofs[0]], u[u_basis.nodal_dofs[0]])
        assert_allclose(ux[uxbasis.facet_dofs[0]], u[u_basis.facet_dofs[0]])
        assert_allclose(uy[uybasis.nodal_dofs[0]], u[u_basis.nodal_dofs[1]])
        assert_allclose(uy[uybasis.facet_dofs[0]], u[u_basis.facet_dofs[1]])


class TestCompositeFacetAssembly(TestCase):

    def runTest(self):

        m = MeshTri()

        fbasis1 = FacetBasis(m, ElementTriP1() * ElementTriP1(),
                             facets=m.facets_satisfying(lambda x: x[0] == 0))
        fbasis2 = FacetBasis(m, ElementTriP1(),
                             facets=lambda x: x[0] == 0)
        fbasis3 = FacetBasis(m, ElementTriP1(), facets='left')

        @BilinearForm
        def uv1(u, p, v, q, w):
            return u * v + p * q

        @BilinearForm
        def uv2(u, v, w):
            return u * v

        A = asm(uv1, fbasis1)
        B = asm(uv2, fbasis2)
        C = asm(uv2, fbasis2)

        assert_allclose(A[0].todense()[0, ::2],
                        B[0].todense()[0])

        assert_allclose(A[0].todense()[0, ::2],
                        C[0].todense()[0])


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

    def prepare_mesh(self):
        return self.mesh_type().refined(self.nrefs)

    def runTest(self):
        m = self.prepare_mesh()
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

    mesh_type = MeshLine1
    element_type = ElementLineP2


class TestInterpolatorLine2(TestInterpolatorTet):

    mesh_type = MeshLine1
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
        (MeshTri, ElementTriP2(), 1, 10),
        (MeshTri, ElementTriP1(), 5, 10),
        (MeshTri, ElementTriP1(), 1, 3e5),
        (MeshTet, ElementTetP2(), 1, 10),
        (MeshTet, ElementTetP1(), 4, 10),
        (MeshTet, ElementTetP1(), 1, 3e4),
        (MeshQuad, ElementQuad1(), 1, 10),
        (MeshQuad, ElementQuad1(), 1, 3e5),
        (MeshHex, ElementHex1(), 1, 1e5),
        (MeshWedge1, ElementWedge1(), 0, 10),
    ]
)
def test_interpolator_probes(mtype, e, nrefs, npoints):

    m = mtype()
    if nrefs > 0:
        m = m.refined(nrefs)

    np.random.seed(0)
    X = np.random.rand(m.p.shape[0], int(npoints))

    basis = CellBasis(m, e)

    y = projection(lambda x: x[0] + x[1], basis)

    assert_allclose(basis.probes(X) @ y, basis.interpolator(y)(X))
    assert_allclose(basis.probes(X) @ y, X[0] + X[1])


@pytest.mark.parametrize(
    "mtype,e1,e2,flat",
    [
        (MeshTri, ElementTriP1(), ElementTriP0(), False),
        (MeshTri, ElementTriP1(), ElementTriP1(), False),
        (MeshTri, ElementTriP2(), ElementTriP1(), False),
        (MeshTri, ElementTriP2(), ElementTriP2(), False),
        (MeshTri, ElementTriP1(), ElementTriP0(), True),
        (MeshTri, ElementTriP1(), ElementTriP1(), True),
        (MeshTri, ElementTriP2(), ElementTriP1(), True),
        (MeshTri, ElementTriP2(), ElementTriP2(), True),
        (MeshTri, ElementTriP2(), None, False),
        (MeshTri, ElementTriP2(), None, True),
        (MeshQuad, ElementQuad1(), ElementQuad0(), False),
        (MeshQuad, ElementQuad1(), ElementQuad1(), False),
        (MeshQuad, ElementQuad2(), ElementQuad2(), False),
        (MeshQuad, ElementQuad1(), ElementQuad0(), True),
        (MeshQuad, ElementQuad1(), ElementQuad1(), True),
        (MeshQuad, ElementQuad2(), ElementQuad2(), True),
        (MeshTet, ElementTetP1(), ElementTetP0(), False),
        (MeshTet, ElementTetP2(), ElementTetP2(), False),
        (MeshHex, ElementHex1(), ElementHex0(), False),
        (MeshHex, ElementHex1(), ElementHex1(), False),
        (MeshHex, ElementHex2(), ElementHex2(), False),
    ],
)
def test_trace(mtype, e1, e2, flat):

    m = mtype().refined(3)

    # use the boundary where last coordinate is zero
    basis = FacetBasis(m, e1, facets=m.facets_satisfying(lambda x: x[-1] == 0.0))
    xfun = projection(lambda x: x[0], CellBasis(m, e1))
    nbasis, y = basis.trace(xfun, lambda p: p[0] if flat else p[:-1], target_elem=e2)

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

    mesh = MeshLine1().refined()
    basis = CellBasis(mesh, etype())
    source = np.array([0.7])
    u = solve(*condense(asm(laplace, basis), basis.point_source(source), D=basis.get_dofs()))
    exact = np.stack([(1 - source) * mesh.p, (1 - mesh.p) * source]).min(0)
    assert_almost_equal(u[basis.nodal_dofs], exact)


def test_pickling():
    # some simple checks for pickle
    mesh = MeshQuad()
    elem = ElementQuad1()
    mapping = MappingIsoparametric(mesh, elem)
    basis = CellBasis(mesh, elem, mapping)

    pickled_mesh = pickle.dumps(mesh)
    pickled_elem = pickle.dumps(elem)
    pickled_mapping = pickle.dumps(mapping)
    pickled_basis = pickle.dumps(basis)

    mesh1 = pickle.loads(pickled_mesh)
    elem1 = pickle.loads(pickled_elem)
    mapping1 = pickle.loads(pickled_mapping)
    basis1 = pickle.loads(pickled_basis)

    assert_almost_equal(
        laplace.assemble(basis).toarray(),
        laplace.assemble(basis1).toarray(),
    )
    assert_almost_equal(
        mesh.doflocs,
        mesh1.doflocs,
    )
    assert_almost_equal(
        mapping.J(0, 0, np.array([[.3], [.3]])),
        mapping1.J(0, 0, np.array([[.3], [.3]])),
    )
    assert_almost_equal(
        elem.doflocs,
        elem1.doflocs,
    )


@pytest.mark.parametrize(
    "m1, m2, lenright",
    [
        (
            MeshTri().refined(3),
            MeshTri().translated((1., 0.)).refined(2),
            1.,
        ),
        (
            MeshTri.init_refdom().refined(3).with_boundaries(
                {'right': lambda x: x[0] + x[1] == 1}
            ),
            MeshTri().translated((1., 0.)).refined(2),
            np.sqrt(2),
        )
    ]
)
def test_mortar_basis(m1, m2, lenright):
    # some sanity checks for MortarBasis
    e = ElementTriP1()

    mort = MappingMortar.init_2D(m1,
                                 m2,
                                 m1.boundaries['right'],
                                 m2.boundaries['left'],
                                 [0., 1.])

    mb = [
        MortarFacetBasis(m1, e, mapping=mort, intorder=4, side=0),
        MortarFacetBasis(m2, e, mapping=mort, intorder=4, side=1),
    ]

    @Functional
    def unity(w):
        return 1.

    @LinearForm
    def load(v, w):
        return 1. * v

    @BilinearForm
    def mass(u, v, w):
        return u * v

    assert_almost_equal(unity.assemble(mb[0]), lenright)
    assert_almost_equal(unity.assemble(mb[1]), lenright)

    assert_almost_equal(load.assemble(mb[0]).dot(m1.p[1]), .5 * lenright)
    assert_almost_equal(load.assemble(mb[1]).dot(m2.p[1]), .5 * lenright)

    # integral is over the domain of the first argument
    assert_almost_equal((mass.assemble(mb[0], mb[1])
                         .dot(m1.p[0] * 0. + 1.)
                         .dot(m2.p[0] * 0. + 1.)), lenright)

    assert_almost_equal((mass.assemble(mb[1], mb[0])
                         .dot(m2.p[0] * 0. + 1.)
                         .dot(m1.p[0] * 0. + 1.)), lenright)

    assert_allclose(mass.assemble(mb[0], mb[1]).toarray(),
                    mass.assemble(mb[1], mb[0]).T.toarray())
