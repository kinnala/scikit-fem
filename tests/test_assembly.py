import sys
from unittest import TestCase

import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal, assert_array_equal)

from skfem import (TrilinearForm, BilinearForm, LinearForm, Functional, asm,
                   solve, condense)
from skfem.element import (ElementQuad1, ElementQuadS2, ElementHex1,
                           ElementHexS2, ElementTetP0, ElementTetP1,
                           ElementTetP2, ElementTriP1, ElementQuad2,
                           ElementTriMorley, ElementVector, ElementQuadP,
                           ElementHex2, ElementTriArgyris, ElementTriP2,
                           ElementDG, ElementTriHermite, ElementTriRT1,
                           ElementTriRT2, ElementTriBDM1,
                           ElementQuadRT1, ElementTetRT1, ElementHexRT1,
                           ElementTriN1, ElementTriP0, ElementTetN1,
                           ElementQuadN1, ElementQuad0, ElementTriN2,
                           ElementTriHHJ1, ElementWedge1,
                           ElementLineP1, ElementLineP1DG)
from skfem.mesh import (MeshQuad, MeshHex, MeshTet, MeshTri, MeshQuad2,
                        MeshTri2, MeshTet2, MeshHex2, MeshWedge, MeshLine)
from skfem.assembly import FacetBasis, Basis, InteriorFacetBasis
from skfem.models import unit_load, mass
from skfem.helpers import grad, dot, ddot, sym_grad, curl, div
from skfem.models import linear_stress


class IntegrateOneOverBoundaryQ1(TestCase):

    elem = ElementQuad1()

    def createBasis(self):
        m = MeshQuad().refined(6)
        self.fbasis = FacetBasis(m, self.elem)
        self.boundary_area = 4.0000

    def runTest(self):
        self.createBasis()

        @BilinearForm
        def uv(u, v, w):
            return u * v

        B = asm(uv, self.fbasis)

        # assemble the same matrix using multiple threads
        if sys.platform != "emscripten":  # no threads in Pyodide
            @BilinearForm(nthreads=2)
            def uvt(u, v, w):
                return u * v

            Bt = asm(uvt, self.fbasis)
        else:
            Bt = B

        @LinearForm
        def gv(v, w):
            return 1.0 * v

        g = asm(gv, self.fbasis)

        ones = np.ones(g.shape)

        self.assertAlmostEqual(ones @ g, self.boundary_area, places=4)
        self.assertAlmostEqual(ones @ (B @ ones), self.boundary_area, places=4)
        self.assertAlmostEqual(ones @ (Bt @ ones), self.boundary_area, places=4)


class IntegrateOneOverBoundaryS2(IntegrateOneOverBoundaryQ1):

    elem = ElementQuadS2()


class IntegrateOneOverBoundaryHex1(IntegrateOneOverBoundaryQ1):

    def createBasis(self):
        m = MeshHex().refined(3)
        self.fbasis = FacetBasis(m, ElementHex1())
        self.boundary_area = 6.000


class IntegrateOneOverBoundaryHex1_2(IntegrateOneOverBoundaryQ1):

    def createBasis(self):
        m = MeshHex().refined(3)
        self.fbasis = FacetBasis(m, ElementHexS2())
        self.boundary_area = 6.000


class IntegrateOneOverBoundaryHex2(IntegrateOneOverBoundaryQ1):

    def createBasis(self):
        m = MeshHex().refined(3)
        self.fbasis = FacetBasis(m, ElementHex2())
        self.boundary_area = 6.000


class IntegrateFuncOverBoundary(TestCase):

    def runTest(self):
        cases = [(MeshHex, ElementHex1),
                 (MeshTet, ElementTetP1),
                 (MeshTet, ElementTetP0)]

        for (mtype, etype) in cases:
            m = mtype().refined(3)
            fb = FacetBasis(m, etype())

            @BilinearForm
            def uv(u, v, w):
                x, y, z = w.x
                return x ** 2 * y ** 2 * z ** 2 * u * v

            B = asm(uv, fb)

            ones = np.ones(B.shape[0])

            self.assertAlmostEqual(ones @ (B @ ones), 0.3333333333, places=5)


class IntegrateFuncOverBoundaryPart(TestCase):

    case = (MeshHex, ElementHex1)

    def runTest(self):
        mtype, etype = self.case
        m = mtype().refined(3)
        bnd = m.facets_satisfying(lambda x: x[0] == 1.0)
        fb = FacetBasis(m, etype(), facets=bnd)

        @BilinearForm
        def uv(u, v, w):
            x, y, z = w.x
            return x ** 2 * y ** 2 * z ** 2 * u * v

        B = asm(uv, fb)
        ones = np.ones(B.shape[0])

        self.assertAlmostEqual(ones @ (B @ ones), 0.11111111, places=5)


class IntegrateFuncOverBoundaryPartHexS2(IntegrateFuncOverBoundaryPart):

    case = (MeshHex, ElementHexS2)


class IntegrateFuncOverBoundaryPartHex2(IntegrateFuncOverBoundaryPart):

    case = (MeshHex, ElementHex2)


class IntegrateFuncOverBoundaryPartTetP1(IntegrateFuncOverBoundaryPart):

    case = (MeshTet, ElementTetP1)


class IntegrateFuncOverBoundaryPartTetP2(IntegrateFuncOverBoundaryPart):

    case = (MeshTet, ElementTetP2)


class IntegrateFuncOverBoundaryPartTetP0(IntegrateFuncOverBoundaryPart):

    case = (MeshTet, ElementTetP0)


class BasisInterpolator(TestCase):

    case = (MeshTri, ElementTriP1)

    def initOnes(self, basis):
        return np.ones(basis.N)

    def runTest(self):
        mtype, etype = self.case
        m = mtype().refined(3)
        e = etype()
        ib = Basis(m, e)

        x = self.initOnes(ib)
        f = ib.interpolator(x)

        X = np.array([np.sin(m.p[0, :]), np.sin(3. * m.p[1, :])])
        self.assertTrue(np.sum(f(X) - 1.0) < 1.0e-10)


class BasisInterpolatorVec(BasisInterpolator):

    case = (MeshTri, lambda: ElementVector(ElementTriP1()))


class BasisInterpolatorTensor(BasisInterpolator):

    def initOnes(self, basis):
        return basis.project(lambda x: np.array([x, x]) * 0 + 1)

    case = (MeshTri, ElementTriHHJ1)


class BasisInterpolatorN1(BasisInterpolator):

    def initOnes(self, basis):
        return basis.project(lambda x: x * 0 + 1)

    case = (MeshTri, ElementTriN1)


class BasisInterpolatorTriP2(BasisInterpolator):

    case = (MeshQuad, ElementQuad1)


class BasisInterpolatorQuad1(BasisInterpolator):

    case = (MeshQuad, ElementQuad1)


class BasisInterpolatorQuad2(BasisInterpolator):

    case = (MeshQuad, ElementQuad2)


class BasisInterpolatorQuadS2(BasisInterpolator):

    case = (MeshQuad, ElementQuadS2)


class BasisInterpolatorMorley(BasisInterpolator):

    case = (MeshTri, ElementTriMorley)

    def initOnes(self, basis):

        @BilinearForm
        def mass(u, v, w):
            return u * v

        @LinearForm
        def ones(v, w):
            return 1. * v

        M = asm(mass, basis)
        f = asm(ones, basis)

        return solve(M, f)


class NormalVectorTestTri(TestCase):

    case = (MeshTri(), ElementTriP1())
    test_integrate_volume = True
    intorder = None

    def runTest(self):
        m = self.case[0].refined()

        if self.intorder is not None:
            basis = FacetBasis(m, self.case[1], intorder=self.intorder)
        else:
            basis = FacetBasis(m, self.case[1])

        @LinearForm
        def linf(v, w):
            return np.sum(w.n ** 2, axis=0) * v

        b = asm(linf, basis)
        ones = basis.project(lambda x: 1.0 + x[0] * 0.)

        self.assertAlmostEqual(b @ ones,
                               2 * m.p.shape[0],
                               places=10)

        if self.test_integrate_volume:
            # by Gauss theorem this integrates to one
            for itr in range(m.p.shape[0]):
                @LinearForm
                def linf(v, w):
                    return w.n[itr] * v

                b = asm(linf, basis)
                self.assertAlmostEqual(b @ m.p[itr, :], 1.0, places=5)


class NormalVectorTestTet(NormalVectorTestTri):

    case = (MeshTet(), ElementTetP1())


class NormalVectorTestTetP2(NormalVectorTestTri):

    case = (MeshTet(), ElementTetP2())
    test_integrate_volume = False


class NormalVectorTestQuad(NormalVectorTestTri):

    case = (MeshQuad(), ElementQuad1())


class NormalVectorTestQuadP(NormalVectorTestTri):

    case = (MeshQuad(), ElementQuadP(3))
    test_integrate_volume = False


class NormalVectorTestHex(NormalVectorTestTri):

    case = (MeshHex(), ElementHex1())
    intorder = 3


class NormalVectorTestHexS2(NormalVectorTestTri):

    case = (MeshHex(), ElementHexS2())
    intorder = 3
    test_integrate_volume = False


class NormalVectorTestHex2(NormalVectorTestTri):

    case = (MeshHex(), ElementHex2())
    intorder = 3
    test_integrate_volume = False


@pytest.mark.parametrize(
    "mtype,e,mtype2",
    [
        (MeshTri, ElementTriP1(), None),
        (MeshTri, ElementTriArgyris(), None),
        (MeshHex, ElementHex1(), None),
        (MeshQuad, ElementQuad1(), None),
        (MeshQuad, ElementQuad2(), None),
        (MeshQuad, ElementQuad2(), MeshQuad2),
        (MeshTri, ElementTriP1(), MeshTri2),
        (MeshTet, ElementTetP1(), MeshTet2),
        (MeshHex, ElementHex1(), MeshHex2),
    ]
)
def test_evaluate_functional(mtype, e, mtype2):
        m = mtype().refined(3)
        if mtype2 is not None:
            m = mtype2.from_mesh(m)
        basis = Basis(m, e)

        @Functional
        def x_squared(w):
            return w.x[0] ** 2

        y = asm(x_squared, basis)

        assert_almost_equal(y, 1. / 3.)
        assert_equal(len(x_squared.elemental(basis)),
                     m.t.shape[1])


class TestRefinterp(TestCase):

    def runTest(self):
        m = MeshQuad().refined(2)
        e = ElementQuad1()
        basis = Basis(m, e)

        M, X = basis.refinterp(m.p[0], 3)

        self.assertEqual(M.p.shape[1], len(X))


class TestCompositeAssembly(TestCase):

    def runTest(self):

        m = MeshHex()
        # check that these assemble to the same matrix
        ec = ElementHex1() * ElementHex1() * ElementHex1()
        ev = ElementVector(ElementHex1())
        basisc = Basis(m, ec)
        basisv = Basis(m, ev)

        @BilinearForm
        def bilinf_ev(u, v, w):
            from skfem.helpers import dot
            return dot(u, v)

        @BilinearForm
        def bilinf_ec(ux, uy, uz, vx, vy, vz, w):
            return ux * vx + uy * vy + uz * vz

        Kv = asm(bilinf_ev, basisv)
        Kc = asm(bilinf_ec, basisc)

        self.assertAlmostEqual(np.sum(np.sum((Kv - Kc).todense())), 0.)


class TestFieldInterpolation(TestCase):

    def runTest(self):

        m = MeshTri()
        e = ElementTriP1()
        basis = Basis(m, e)

        @Functional
        def feqx(w):
            from skfem.helpers import grad
            f = w['func']  # f(x) = x
            return grad(f)[0]  # f'(x) = 1

        func = basis.interpolate(m.p[0])

        # integrate f'(x) = 1 over [0, 1]^2
        self.assertAlmostEqual(feqx.assemble(basis, func=func), 1.)


class TestFieldInterpolation_2(TestCase):

    def runTest(self):

        m = MeshTri()
        e = ElementTriP1()
        basis = Basis(m, e)

        @Functional
        def feqx(w):
            from skfem.helpers import grad
            f = w['func']  # f(x, y) = x
            g = w['gunc']  # g(x, y) = y
            return grad(f)[0] + grad(g)[1]

        func = basis.interpolate(m.p[0])
        gunc = basis.interpolate(m.p[1])

        self.assertAlmostEqual(feqx.assemble(basis, func=func, gunc=gunc), 2.)


class VectorialFunctional(TestCase):

    def runTest(self):

        @Functional
        def hydrostatic_pressure(w):
            return w.n * w.x[1]

        np.testing.assert_allclose(
            hydrostatic_pressure.assemble(FacetBasis(MeshTri(), ElementTriP1())), [0, 1]
        )


class TestComplexValuedAssembly(TestCase):

    def runTest(self):

        m = MeshTri()
        e = ElementTriP1()
        basis = Basis(m, e)
        self.interior_area = 1

        @BilinearForm(dtype=np.complex64)
        def complexmass(u, v, w):
            return 1j * u * v

        @LinearForm(dtype=np.complex64)
        def complexfun(v, w):
            return 1j * v

        M = asm(complexmass, basis)
        f = asm(complexfun, basis)
        ones = np.ones(M.shape[1])

        self.assertAlmostEqual(np.dot(ones, M @ ones), 1j * self.interior_area)
        self.assertAlmostEqual(np.dot(ones, f), 1j * self.interior_area)


@pytest.mark.skipif(sys.platform == "emscripten",
                    reason="threads are not available in Pyodide")
class TestThreadedAssembly(TestCase):

    def runTest(self):

        m = MeshTri().refined()
        e = ElementTriP1()
        basis = Basis(m, e)

        @BilinearForm
        def nonsym(u, v, w):
            return u.grad[0] * v

        @BilinearForm(nthreads=2)
        def threaded_nonsym(u, v, w):
            return u.grad[0] * v

        assert_almost_equal(
            nonsym.assemble(basis).toarray(),
            threaded_nonsym.assemble(basis).toarray(),
        )


@pytest.mark.parametrize(
    "m,e,edg",
    [
        (MeshTri().refined(), ElementTriP1(), ElementDG),
        (MeshTri().refined(), ElementTriP2(), ElementDG),
        (MeshTet().refined(), ElementTetP1(), ElementDG),
        (MeshTet().refined(), ElementTetP2(), ElementDG),
        (MeshTri().refined(), ElementTriMorley(), ElementDG),
        (MeshTri().refined(), ElementTriHermite(), ElementDG),
        (MeshQuad().refined(), ElementQuad1(), ElementDG),
        (MeshQuad().refined(), ElementQuad2(), ElementDG),
        (MeshQuad().refined(), ElementQuadP(4), ElementDG),
        (MeshHex().refined(), ElementHex2(), ElementDG),
    ]
)
def test_coodata_inverse(m, e, edg):

    E = edg(e)
    basis = Basis(m, E)
    basisdg = Basis(m, E)
    M1 = mass.assemble(basis)
    M2 = mass.elemental(basisdg)
    assert_array_almost_equal(
        np.linalg.inv(M1.toarray()),
        M2.inverse().tocsr().toarray(),
    )


def test_coodata_tolocal_bilinear():
    # nonsymmetric form with different trial and test spaces; DG spaces so
    # that the local matrices are the diagonal blocks of the global matrix
    m = MeshTri().refined()
    ubasis = Basis(m, ElementDG(ElementTriP2()), intorder=4)
    vbasis = Basis(m, ElementDG(ElementTriP1()), intorder=4)
    A = (BilinearForm(lambda u, v, w: grad(u)[0] * v)
         .elemental(ubasis, vbasis))
    K = A.toarray()
    local = A.tolocal()
    assert local.shape == (m.nelements, vbasis.Nbfun, ubasis.Nbfun)
    for e in range(m.nelements):
        assert_array_almost_equal(
            local[e],
            K[np.ix_(vbasis.element_dofs[:, e], ubasis.element_dofs[:, e])],
        )
    assert_array_almost_equal(A.fromlocal(local).toarray(), K)


def test_coodata_tolocal_trilinear():
    m = MeshTri()
    ubasis = Basis(m, ElementDG(ElementTriP2()), intorder=4)
    vbasis = Basis(m, ElementDG(ElementTriP1()), intorder=4)
    wbasis = Basis(m, ElementDG(ElementTriP0()), intorder=4)
    A = (TrilinearForm(lambda u, v, w, p: w * grad(u)[0] * v)
         .elemental(ubasis, vbasis, wbasis))
    K = A.toarray()
    local = A.tolocal()
    assert local.shape == (m.nelements,
                           wbasis.Nbfun,
                           vbasis.Nbfun,
                           ubasis.Nbfun)
    for e in range(m.nelements):
        assert_array_almost_equal(
            local[e],
            K[np.ix_(wbasis.element_dofs[:, e],
                     vbasis.element_dofs[:, e],
                     ubasis.element_dofs[:, e])],
        )
    assert_array_almost_equal(A.fromlocal(local).toarray(), K)


def test_coodata_inverse_nonsymmetric():
    m = MeshTri().refined()
    basis = Basis(m, ElementDG(ElementTriP1()))
    A = BilinearForm(lambda u, v, w: u * v + grad(u)[0] * v).elemental(basis)
    assert_array_almost_equal(
        np.linalg.inv(A.toarray()),
        A.inverse().toarray(),
    )


def test_multimesh():

    m1 = MeshTri().refined()
    m2 = MeshQuad().refined().translated((1., 0.))
    m3 = MeshTri().refined().translated((2., 0.))
    M = m1 @ m2 @ m3

    assert len(M) == 3

    E = [ElementTriP1(), ElementQuad1(), ElementTriP1()]
    basis = list(map(Basis, M, E))

    Mm = asm(mass, basis)

    assert Mm.shape[0] == 3 * 7


def test_multimesh_2():

    m = MeshTri()
    m1 = MeshTri.init_refdom()
    m2 = MeshTri.init_refdom().scaled((-1., -1.)).translated((1., 1.,))
    M = m1 @ m2

    E = [ElementTriP1(), ElementTriP1()]
    basis1 = list(map(Basis, M, E))
    basis2 = Basis(m, ElementTriP1())

    M1 = asm(mass, basis1)
    M2 = asm(mass, basis2)

    assert_array_almost_equal(M1.toarray(), M2.toarray())


@pytest.mark.parametrize(
    "m,e",
    [
        (MeshTri().refined(), ElementTriP1()),
        (MeshTri(), ElementTriP2()),
        (MeshTet(), ElementTetP1()),
        (MeshQuad().refined(), ElementQuad1()),
        (MeshHex(), ElementHex2()),
    ]
)
def test_trilinear_form(m, e):

    basis = Basis(m, e)
    out = (TrilinearForm(lambda u, v, w, p: dot(w * grad(u), grad(v)))
           .assemble(basis))
    kp = np.random.rand(100, basis.N)

    # transform to a dense 3-tensor (slow)
    A = out.toarray()
    # # initialize a sparse tensor instead
    # import sparse
    # arr = sparse.COO(*out.astuple(), has_duplicates=True)

    opt1 = np.einsum('ijk,li->ljk', A, kp)

    for i in range(kp.shape[0]):
        opt2 = (BilinearForm(lambda u, v, p: dot(p['kp'] * grad(u), grad(v)))
                .assemble(basis, kp=kp[i]))
        assert abs((opt1[i] - opt2).min()) < 1e-10
        assert abs((opt1[i] - opt2).max()) < 1e-10


@pytest.mark.parametrize(
    "m,e",
    [
        (MeshTri(), ElementTriP1()),
        (MeshTri(), ElementTriP2()),
        (MeshTet(), ElementTetP1()),
    ]
)
def test_matrix_element_projection(m, e):

    E1 = ElementVector(e)
    E2 = ElementVector(ElementVector(e))
    basis0 = Basis(m, E1)
    basis1 = basis0.with_element(E2)
    C = linear_stress()

    x = basis0.interpolate(np.random.random(basis0.N))

    basis1.project(C(sym_grad(x)))


@pytest.mark.parametrize(
    "basis",
    [
        Basis(MeshTri().refined(6).with_defaults(),
              ElementTriRT1()),
        Basis(MeshTri().refined(4).with_defaults(),
              ElementTriRT2()),
        Basis(MeshTri().refined(5).with_defaults(),
              ElementTriBDM1()),
        Basis(MeshQuad().refined(4).with_defaults(),
              ElementQuadRT1()),
    ]
)
def test_hdiv_boundary_integration(basis):

    y = basis.project(lambda x: np.array([x[0], 0 * x[1]]))
    fbasis = basis.boundary('right')

    @Functional
    def test1(w):
        return dot(w['y'], w.n)

    @Functional
    def test2(w):
        return w['y'].div

    assert_almost_equal(test1.assemble(fbasis, y=y), 1, decimal=2)

    if not isinstance(basis.elem, ElementTriRT1):
        assert_almost_equal(test2.assemble(fbasis, y=y), 1, decimal=5)


@pytest.mark.parametrize(
    "basis",
    [
        Basis(MeshTet().refined(3).with_defaults(),
              ElementTetRT1()),
        Basis(MeshHex().refined(3).with_defaults(),
              ElementHexRT1()),
    ]
)
def test_hdiv_boundary_integration_3d(basis):

    y = basis.project(lambda x: np.array([x[0], 0 * x[1], 0 * x[2]]))
    fbasis = basis.boundary('right')

    @Functional
    def test1(w):
        return dot(w['y'], w.n)

    @Functional
    def test2(w):
        return w['y'].div

    assert_almost_equal(test1.assemble(fbasis, y=y), 1, decimal=1)
    if not isinstance(basis.elem, ElementTetRT1):
        assert_almost_equal(test2.assemble(fbasis, y=y), 1, decimal=5)


@pytest.mark.parametrize(
    "basis",
    [
        Basis(MeshTri().refined(4).with_defaults(),
              ElementTriN1()),
        Basis(MeshTri().refined(4).with_defaults(),
              ElementTriN2()),
        Basis(MeshQuad().refined(3).with_defaults(),
              ElementQuadN1()),
    ]
)
def test_hcurl_boundary_integration(basis):

    y = basis.project(lambda x: np.array([x[1], -x[0]]))
    fbasis = basis.boundary('right')

    @Functional
    def test1(w):
        rotx = np.array([w.x[1], -w.x[0]])
        return dot(w['y'], w.n) - dot(rotx, w.n)

    @Functional
    def test2(w):
        return w['y'].curl

    assert_almost_equal(test1.assemble(fbasis, y=y), 0, decimal=2)
    assert_almost_equal(test2.assemble(fbasis, y=y), -2, decimal=5)


@pytest.mark.parametrize(
    "basis",
    [
        Basis(MeshTet().refined(2).with_defaults(),
              ElementTetN1()),
    ]
)
def test_hcurl_boundary_integration_3d(basis):

    y = basis.project(lambda x: np.array([x[1], -x[0], 0*x[0]]))
    fbasis = basis.boundary('right')

    @Functional
    def test1(w):
        rotx = np.array([w.x[1], -w.x[0], w.x[2]*0.])
        return dot(w['y'], w.n) - dot(rotx, w.n)

    @Functional
    def test2(w):
        return w['y'].curl

    assert_almost_equal(test1.assemble(fbasis, y=y), 0, decimal=2)
    assert_almost_equal(test2.assemble(fbasis, y=y),
                        np.array([0, 0, -2]),
                        decimal=5)


@pytest.mark.parametrize(
    "mtype,e1,e2,e3",
    [
        (MeshTri, ElementVector(ElementTriP1()), ElementTriN1(), ElementTriP0()),
        (MeshTri, ElementVector(ElementTriP2()), ElementTriN2(), ElementTriP0()),
        (MeshQuad, ElementVector(ElementQuad1()), ElementQuadN1(), ElementQuad0()),
    ]
)
def test_hcurl_projections_2d(mtype, e1, e2, e3):

    m = mtype.init_tensor(np.linspace(-1, 1, 20),
                          np.linspace(-1, 1, 20))
    basis0 = Basis(m, e1)
    y = basis0.project(lambda x: np.array([x[1], -x[0]]))
    basis = basis0.with_element(e2)
    x = basis.project(lambda x: np.array([x[1], -x[0]]))
    dbasis = basis0.with_element(e3)
    curla = dbasis.project(lambda x: x[0] * 0 - 2.)

    curly = dbasis.project(curl(basis0.interpolate(y)))
    curlx = dbasis.project(curl(basis.interpolate(x)))

    assert_almost_equal(curly, curlx)
    assert_almost_equal(curly, curla)


def test_hcurl_projections_3d():

    m = MeshTet.init_tensor(np.linspace(-1, 1, 10),
                            np.linspace(-1, 1, 10),
                            np.linspace(-1, 1, 10))
    basis0 = Basis(m, ElementVector(ElementTetP1()))
    y = basis0.project(lambda x: np.array([x[1], -x[0], 0*x[0]]))
    basis = basis0.with_element(ElementTetN1())
    x = basis.project(lambda x: np.array([x[1], -x[0], 0*x[0]]))
    dbasis = basis0.with_element(ElementVector(ElementTetP0()))
    curla = dbasis.project(lambda x: np.array([0*x[0], 0*x[0], x[0] * 0 - 2.]))

    curly = dbasis.project(curl(basis0.interpolate(y)))
    curlx = dbasis.project(curl(basis.interpolate(x)))

    assert_almost_equal(curly, curlx)
    assert_almost_equal(curly, curla)


def test_element_global_boundary_normal():

    mesh = MeshTri.init_sqsymmetric().refined(3)
    basis = Basis(mesh, ElementTriMorley())

    D = basis.get_dofs().all('u_n')
    x = basis.zeros()
    x[D] = 1

    @BilinearForm
    def bilinf(u, v, w):
        return ddot(u.hess, v.hess)

    A = bilinf.assemble(basis)
    f = 0 * unit_load.assemble(basis)

    y = solve(*condense(A, f, D=basis.get_dofs(), x=x))

    assert (y[basis.get_dofs(elements=True).all('u')] <= 0).all()


@pytest.mark.parametrize(
    "basis",
    [
        Basis(MeshTri().refined(6), ElementTriN1()),
        Basis(MeshTri().refined(2),
              ElementVector(ElementTriP1())),
    ]
)
def test_vectorial_interpolator(basis):

    m = basis.mesh
    y = basis.project(lambda x: x)
    yfun = basis.interpolator(y)

    X = np.array([np.sin(m.p[0, :]), np.sin(3. * m.p[1, :])])
    assert_almost_equal(yfun(X), X, decimal=2)


@pytest.mark.parametrize(
    "basis",
    [
        Basis(MeshTri().refined(2),
              ElementVector(ElementVector(ElementTriP1()))),
        Basis(MeshTri().refined(2),
              ElementTriHHJ1()),
    ]
)
def test_tensorial_interpolator(basis):

    m = basis.mesh
    y = basis.project(lambda x: np.array([[x[0], x[1]],
                                          [x[1], x[0]]]))
    yfun = basis.interpolator(y)

    X = np.array([np.sin(m.p[0, :]), np.sin(3. * m.p[1, :])])
    assert_almost_equal(yfun(X), np.array([[X[0], X[1]],
                                           [X[1], X[0]]]), decimal=4)


@pytest.mark.parametrize(
    "mesh,element,target",
    [
        (MeshLine(), ElementLineP1(), 1.),
        (MeshTri(), ElementTriP1(), 1.),
        (MeshQuad(), ElementQuad1(), 1.),
        (MeshTet(), ElementTetP1(), 1.),
        (MeshHex(), ElementHex1(), 1.),
        (MeshTri2(), ElementTriP1(), 1.),
        (MeshQuad2(), ElementQuad1(), 1.),
        (MeshWedge(), ElementWedge1(), 1.),
    ]
)
def test_integrate_unit_volume(mesh, element, target):

    basis = Basis(mesh, element)
    assert_almost_equal(
        Functional(lambda w: 1. + 0. * w.x[0]).assemble(basis),
        target
    )


@pytest.mark.parametrize(
    "facet_basis",
    [
        lambda m, e: FacetBasis(m, e),
        lambda m, e: InteriorFacetBasis(m, e, side=0),
        lambda m, e: InteriorFacetBasis(m, e, side=1),
        lambda m, e: Basis(m, e).skeleton(ElementLineP1DG()).bases[0],
    ]
)
def test_coodata_tolocal_facets(facet_basis):
    # summing the local facet matrices to elements must give the global
    # matrix; with a DG space the element matrices are the diagonal blocks
    m = MeshTri().refined(2)
    e = ElementDG(ElementTriP1())
    fb = facet_basis(m, e)
    form = BilinearForm(lambda u, v, w: u * v + dot(w.n, grad(u)) * v)
    local = form.elemental(fb).tolocal(fb)
    assert local.shape == (m.nelements, fb.Nbfun, fb.Nbfun)
    lap = BilinearForm(lambda u, v, w: 0. * u * v).elemental(Basis(m, e))
    assert_array_almost_equal(lap.fromlocal(local).toarray(),
                              form.assemble(fb).toarray())


def test_coodata_tolocal_basis_mismatch():
    m = MeshTri().refined(2)  # different number of boundary/interior facets
    e = ElementDG(ElementTriP1())
    A = BilinearForm(lambda u, v, w: u * v).elemental(FacetBasis(m, e))
    with pytest.raises(ValueError):
        A.tolocal(InteriorFacetBasis(m, e))


def _stokes_basis(m):
    return Basis(m, ElementVector(ElementTriP2()) * ElementTriP1())


@BilinearForm
def _stokes(u, p, v, q, w):
    return ddot(grad(u), grad(v)) - div(u) * q - div(v) * p


@pytest.mark.parametrize(
    "form,basis,ncomp",
    [
        (_stokes, _stokes_basis(MeshTri().refined()), 2),
        (BilinearForm(lambda u, v, w: ddot(sym_grad(u), sym_grad(v))),
         Basis(MeshTri().refined(), ElementVector(ElementTriP1())), 2),
        (BilinearForm(lambda u, v, w: ddot(sym_grad(u), sym_grad(v))),
         Basis(MeshTet(), ElementVector(ElementTetP1())), 3),
        (BilinearForm(lambda u, p, q, v, r, s, w: (dot(grad(u), grad(v))
                                                   + p * r + q * s
                                                   + u * s + q * v)),
         Basis(MeshTri().refined(), ElementTriP2() * ElementTriP1()
               * ElementTriP0()), 3),
    ]
)
def test_assemble_blocks(form, basis, ncomp):
    blocks = form.assemble_blocks(basis)
    eblocks = form.elemental_blocks(basis)
    assert len(blocks) == ncomp and all(len(b) == ncomp for b in blocks)
    K = form.assemble(basis)
    ix = basis.split_indices()
    for i in range(ncomp):
        for j in range(ncomp):
            ref = K[ix[i]][:, ix[j]]
            assert blocks[i][j].shape == (len(ix[i]), len(ix[j]))
            assert abs(blocks[i][j] - ref).max() < 1e-12
            assert abs(eblocks[i][j].tocsr() - ref).max() < 1e-12
            assert eblocks[i][j].tolocal().shape[0] == basis.nelems
    # the blocks are numbered as the components given by split
    x = np.arange(basis.N, dtype=np.float64)
    parts = [y for y, _ in basis.split(x)]
    for i in range(ncomp):
        assert_array_almost_equal(sum(blocks[i][j] @ parts[j]
                                      for j in range(ncomp)),
                                  (K @ x)[ix[i]])


def test_assemble_blocks_rectangular():
    m = MeshTri().refined()
    ubasis = Basis(m, ElementTriP1(), intorder=4)
    vbasis = Basis(m, ElementVector(ElementTriP2()) * ElementTriP1(),
                   intorder=4)
    form = BilinearForm(lambda p, v, q, w: div(v) * p + q * p)
    (B,), (M,) = form.assemble_blocks(ubasis, vbasis)
    assert B.shape == (vbasis.split_indices()[0].shape[0], ubasis.N)
    assert M.shape == (vbasis.split_indices()[1].shape[0], ubasis.N)
    (Be,), (Me,) = form.elemental_blocks(ubasis, vbasis)
    assert abs(Be.tocsr() - B).max() < 1e-12
    assert abs(Me.tocsr() - M).max() < 1e-12


def test_linear_assemble_blocks():
    basis = _stokes_basis(MeshTri().refined())
    form = LinearForm(lambda v, q, w: v[1] + q)
    fu, fp = form.assemble_blocks(basis)
    f = form.assemble(basis)
    ix = basis.split_indices()
    assert_array_almost_equal(fu, f[ix[0]])
    assert_array_almost_equal(fp, f[ix[1]])


def test_elemental_blocks_inverse():
    # inverse of a block of the elemental matrices of a DG space
    m = MeshTri().refined()
    basis = Basis(m, ElementDG(ElementTriP1()) * ElementDG(ElementTriP0()))
    form = BilinearForm(lambda u, p, v, q, w: u * v + p * q + u * q)
    (A, _), _ = form.elemental_blocks(basis)
    Ainv = A.inverse().tocsr()
    assert_array_almost_equal((Ainv @ A.tocsr()).toarray(),
                              np.eye(A.shape[0]))


def test_composite_basis_split_indices():
    m = MeshTri().refined()
    b1 = Basis(m, ElementTriP1(), intorder=4)
    b2 = Basis(m, ElementTriP2(), intorder=4)
    cb = b1 * b2
    ix = cb.split_indices()
    x = np.arange(cb.N)
    for (y, _), i in zip(cb.split(x), ix):
        assert_array_equal(y, x[i])
    assert cb.split_bases() == [b1, b2]
    assert [b for _, b in cb.split(x)] == cb.split_bases()
    # the components share the DOFs
    form = BilinearForm(lambda u, ut, v, vt, w: u * v)
    with pytest.raises(ValueError):
        form.assemble_blocks(b1 @ Basis(m, ElementTriP1()))
