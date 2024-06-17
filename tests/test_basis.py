import pickle
from unittest import TestCase
from pathlib import Path

import pytest
import numpy as np
from numpy.testing import (assert_allclose, assert_almost_equal,
                           assert_array_equal)

from skfem import BilinearForm, LinearForm, asm, solve, condense, projection
from skfem.mesh import (Mesh, MeshTri, MeshTet, MeshHex,
                        MeshQuad, MeshLine1, MeshWedge1)
from skfem.assembly import (CellBasis, FacetBasis, Dofs, Functional,
                            InteriorFacetBasis)
from skfem.mapping import MappingIsoparametric
from skfem.element import (ElementVectorH1, ElementTriP2, ElementTriP1,
                           ElementTetP2, ElementHexS2, ElementHex2,
                           ElementQuad2, ElementLineP2, ElementTriP0,
                           ElementLineP0, ElementQuad1, ElementQuad0,
                           ElementTetP1, ElementTetP0, ElementHex1,
                           ElementHex0, ElementLineP1, ElementLineMini,
                           ElementWedge1, ElementTriRT0, ElementQuadRT0,
                           ElementTriP1)
from skfem.helpers import dot, grad
from skfem.utils import enforce
from skfem.io.meshio import to_meshio, from_meshio
from skfem.models.poisson import laplace


MESH_PATH = Path(__file__).parents[1] / 'docs' / 'examples' / 'meshes'


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

        D = basis.get_dofs(skip=['u^2'])
        Dup = basis.get_dofs('up', skip=['u^2'])
        x = basis.zeros()
        x[Dup.all('u^1^1')] = .1

        x = solve(*condense(S, x=x, D=D))

        (u, u_basis), (p, p_basis) = basis.split(x)

        self.assertEqual(len(u), m.p.shape[1] * 2 + m.facets.shape[1] * 2)
        self.assertEqual(len(p), m.p.shape[1])

        self.assertTrue(np.sum(p - x[basis.nodal_dofs[2]]) < 1e-8)

        U, P = basis.interpolate(x)
        self.assertTrue(isinstance(U.value, np.ndarray))
        self.assertTrue(isinstance(P.value, np.ndarray))
        self.assertTrue(P.shape[0] == m.nelements)

        self.assertTrue((basis.doflocs[:, Dup.all()][1] == 1.).all())

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

        m = MeshTri().with_defaults()

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

        y = fbasis1.zeros()
        y[fbasis1.get_dofs('left')] = 1
        assert_allclose(
            fbasis1.project(fbasis1.interpolate(fbasis1.ones())),
            y,
        )


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
            arr1 = basis.get_dofs({
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
    "m, e, fun",
    [
        (MeshTri(), ElementTriP1(), lambda x: x[0]),
        (MeshTri(), ElementTriRT0(), lambda x: x),
        (MeshQuad(), ElementQuadRT0(), lambda x: x),
    ]
)
def test_basis_project(m, e, fun):

    basis = CellBasis(m, e)
    y = basis.project(fun)

    @Functional
    def int_y_1(w):
        return w.y ** 2

    @Functional
    def int_y_2(w):
        return fun(w.x) ** 2

    assert_almost_equal(int_y_1.assemble(basis, y=y),
                        int_y_2.assemble(basis))



@pytest.mark.parametrize(
    "m, e, p1, p2",
    [
        (
            MeshTri(),
            ElementTriP1() * ElementTriP1(),
            (1, 1),
            np.ones(8),
        ),
        (
            MeshTri(),
            ElementTriP1() * ElementTriP1(),
            (lambda x: x[0] * 0 + 1, 1),
            np.ones(8),
        ),
        (
            MeshTri(),
            ElementTriP1() * ElementTriP1(),
            lambda x: (x[0] * 0 + 1, x[0] * 0 + 1),
            np.ones(8),
        ),
        (
            MeshTri(),
            ElementTriP1() * ElementTriP1(),
            lambda x: [x[0] * 0 + 1, x[0] * 0 + 1],
            np.ones(8),
        ),
        (
            MeshTri(),
            ElementTriP1() * ElementTriP1(),
            lambda x: [x[0] * 0 + 1, 1],
            np.ones(8),
        ),
    ]
)
def test_basis_project_composite(m, e, p1, p2):

    basis = CellBasis(m, e)
    y = basis.project(p1)
    assert_almost_equal(y, p2)


def test_basis_project_grad():

    m, e = (MeshTri(), ElementTriP1())
    basis = CellBasis(m, e)
    Y = basis.interpolate(m.p[0])
    y = basis.project(Y.grad[0])

    @Functional
    def int_y(w):
        return w.y ** 2

    assert_almost_equal(int_y.assemble(basis, y=y), 1.)


@pytest.mark.parametrize(
    "m, e",
    [
        (MeshTri(), ElementTriP1()),
        (MeshTri(), ElementTriRT0()),
        (MeshQuad(), ElementQuadRT0()),
        (MeshTri(), ElementTriP2() * ElementTriP1()),
    ]
)
def test_basis_interpolate_project(m, e):

    basis = CellBasis(m, e)
    x = basis.zeros() + 1
    Y = basis.interpolate(x)
    X = basis.project(Y)
    # check that interpolate and project are inverses
    assert_almost_equal(x, X)


def test_subdomain_facet_assembly():

    def subdomain(x):
        return np.logical_and(
            np.logical_and(x[0]>.25, x[0]<.75),
            np.logical_and(x[1]>.25, x[1]<.75),
        )

    m, e = MeshTri().refined(4), ElementTriP2()
    cbasis = CellBasis(m, e)
    cbasis_p0 = cbasis.with_element(ElementTriP0())

    sfbasis = FacetBasis(m, e, facets=m.facets_around(subdomain, flip=True))
    sfbasis_p0 = sfbasis.with_element(ElementTriP0())
    sigma = cbasis_p0.zeros() + 1

    @BilinearForm
    def laplace(u, v, w):
        return dot(w.sigma * grad(u), grad(v))

    A = laplace.assemble(cbasis, sigma=cbasis_p0.interpolate(sigma))
    u0 = cbasis.zeros()
    u0[cbasis.get_dofs(elements=subdomain)] = 1
    u0_dofs = cbasis.get_dofs() + cbasis.get_dofs(elements=subdomain)
    A, b = enforce(A, D=u0_dofs, x=u0)
    u = solve(A, b)

    @Functional
    def measure_current(w):
        return dot(w.n, w.sigma * grad(w.u))

    meas = measure_current.assemble(sfbasis,
                                    sigma=sfbasis_p0.interpolate(sigma),
                                    u=sfbasis.interpolate(u))

    assert_almost_equal(meas, 9.751915526759191)


def test_subdomain_facet_assembly_2():

    m = MeshTri().refined(4).with_subdomains({'all': lambda x: x[0] * 0 + 1})
    e = ElementTriP1()

    @Functional
    def boundary_integral(w):
        return dot(w.n, grad(w.u))

    sfbasis = FacetBasis(m, e, facets=m.facets_around('all'))
    fbasis = FacetBasis(m, e)

    assert_almost_equal(
        boundary_integral.assemble(sfbasis, u=m.p[0] * m.p[1]),
        boundary_integral.assemble(fbasis, u=m.p[0] * m.p[1]),
    )


@pytest.mark.parametrize(
    "m, e, facets, fun",
    [
        (
            MeshTri.load(MESH_PATH / 'interface.msh'),
            ElementTriP1(),
            'interfacee',
            lambda m: m.p[1],
        ),
        (
            MeshTet.load(MESH_PATH / 'cuubat.msh'),
            ElementTetP1(),
            'interface',
            lambda m: m.p[0],
        )
    ]
)
def test_oriented_interface_integral(m, e, facets, fun):

    fb = FacetBasis(m, e, facets=facets)
    assert_almost_equal(
        Functional(lambda w: dot(w.fun.grad, w.n)).assemble(fb, fun=fun(m)),
        1.0,
    )


@pytest.mark.parametrize(
    "m, e, facets, normal",
    [
        (
            MeshTri().refined(2) + MeshTri().translated((1.0, 0.0)).refined(2),
            ElementTriP1(),
            lambda x: x[0] == 1.0,
            np.array([1, 0]),
        ),
        (
            MeshHex().refined(2) + MeshHex().translated((1., 0., 0.)).refined(2),
            ElementHex1(),
            lambda x: x[0] == 1.0,
            np.array([1, 0, 0]),
        ),
    ]
)
def test_oriented_interface_integral2(m, e, facets, normal):

    fb = FacetBasis(m, e, facets=m.facets_satisfying(facets, normal=normal))
    assert_almost_equal(
        Functional(lambda w: dot(w.fun.grad, w.n)).assemble(fb, fun=m.p[0]),
        1.0,
    )


@pytest.mark.parametrize(
    "m, e",
    [
        (
            MeshTri().refined(6).with_subdomains({
                'mid': lambda x: (x[0] - .5) ** 2 + (x[1] - .5) ** 2 < .5 ** 2,
            }),
            ElementTriP1(),
        ),
        (
            MeshTet().refined(4).with_subdomains({
                'mid': lambda x: ((x[0] - .5) ** 2
                                  + (x[1] - .5) ** 2
                                  + (x[2] - .5) ** 2) < .5 ** 2,
            }),
            ElementTetP1(),
        ),
        (
            MeshHex().refined(4).with_subdomains({
                'mid': lambda x: ((x[0] - .5) ** 2
                                  + (x[1] - .5) ** 2
                                  + (x[2] - .5) ** 2) < .5 ** 2,
            }),
            ElementHex1(),
        ),
    ]
)
def test_oriented_gauss_integral(m, e):

    facets = m.facets_around('mid')
    fb = FacetBasis(m, e, facets=facets)
    cb = CellBasis(m, e, elements='mid')
    assert_almost_equal(
        Functional(lambda w: w.x[0] * w.n[0]).assemble(fb),
        Functional(lambda w: 1. + 0. * w.x[0]).assemble(cb),
        decimal=5,
    )

@pytest.mark.parametrize(
    "m", [MeshLine1(), MeshTri(), MeshQuad(), MeshTet(), MeshHex()]
)
def test_oriented_saveload(m: Mesh):

    m = m.refined(4)
    m = m.with_boundaries({"mid": m.facets_around([5]),})
    assert len(m.boundaries["mid"].ori) == m.refdom.nfacets

    M = from_meshio(to_meshio(m))

    assert_array_equal(
        m.boundaries["mid"].ori, M.boundaries["mid"].ori,
    )


class TestZerosOnes(TestCase):

    def runTest(self):
        basis = CellBasis(MeshTri(), ElementTriP0())
        a = basis.zeros(dtype=int) + 1
        b = basis.ones(dtype=float) * 2.
        self.assertEqual(len(b), basis.N)
        self.assertTrue(a.dtype == int)
        self.assertTrue(b.dtype == float)
        assert_array_equal(
            a.astype(float) + b, np.array([3., 3.])
        )


def test_with_elements():
    mesh = MeshTri().refined(3).with_subdomains({'a': lambda x: x[0] < .5})
    basis = CellBasis(mesh, ElementTriP0())
    basis_half = basis.with_elements('a')

    assert basis.mesh == basis_half.mesh
    assert basis.elem == basis_half.elem
    assert basis.mapping == basis_half.mapping
    assert basis.quadrature == basis_half.quadrature
    assert all(basis_half.tind == basis.mesh.normalize_elements('a'))


def test_disable_doflocs():
    mesh = MeshTri().refined(3)
    basis = CellBasis(mesh, ElementTriP1())
    basisd = CellBasis(mesh, ElementTriP1(), disable_doflocs=True)
    fbasis = FacetBasis(mesh, ElementTriP1())
    fbasisd = FacetBasis(mesh, ElementTriP1(), disable_doflocs=True)
    ifbasis = InteriorFacetBasis(mesh, ElementTriP1())
    ifbasisd = InteriorFacetBasis(mesh, ElementTriP1(),
                                  disable_doflocs=True)
    assert not hasattr(fbasisd, 'doflocs')
    assert hasattr(fbasis, 'doflocs')
    assert not hasattr(basisd, 'doflocs')
    assert hasattr(basis, 'doflocs')
    assert not hasattr(ifbasisd, 'doflocs')
    assert hasattr(ifbasis, 'doflocs')
