from unittest import TestCase, main

import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal,
                           assert_array_almost_equal)

from skfem import BilinearForm, LinearForm, Functional, asm, solve, condense
from skfem.element import (ElementQuad1, ElementQuadS2, ElementHex1,
                           ElementHexS2, ElementTetP0, ElementTetP1,
                           ElementTetP2, ElementTriP1, ElementQuad2,
                           ElementTriMorley, ElementVectorH1, ElementQuadP,
                           ElementHex2, ElementTriArgyris, ElementTriP2)
from skfem.mesh import (MeshQuad, MeshHex, MeshTet, MeshTri, MeshQuad2,
                        MeshTri2, MeshTet2, MeshHex2, MeshTri1DG, MeshQuad1DG)
from skfem.assembly import FacetBasis, Basis
from skfem.utils import projection
from skfem.models import laplace, unit_load, mass


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
        @BilinearForm(nthreads=2)
        def uvt(u, v, w):
            return u * v

        Bt = asm(uvt, self.fbasis)

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
        ones = projection(lambda x: 1.0 + x[0] * 0.,
                          basis,
                          I=basis.get_dofs().flatten(),
                          expand=True)

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
        ev = ElementVectorH1(ElementHex1())
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
    "m,mdgtype,etype,check1,check2",
    [
        (
            MeshTri.init_tensor(np.linspace(0, 1, 7),
                                np.linspace(0, 1, 7)),
            MeshTri1DG,
            ElementTriP1,
            lambda x: x[0] == 1,
            lambda x: x[0] == 0,
        ),
        (
            MeshTri.init_tensor(np.linspace(0, 1, 5),
                                np.linspace(0, 1, 5)),
            MeshTri1DG,
            ElementTriP1,
            lambda x: x[0] == 1,
            lambda x: x[0] == 0,
        ),
        (
            MeshTri().refined(2),
            MeshTri1DG,
            ElementTriP1,
            lambda x: x[0] == 1,
            lambda x: x[0] == 0,
        ),
        (
            MeshTri().refined(3),
            MeshTri1DG,
            ElementTriP1,
            lambda x: x[0] == 1,
            lambda x: x[0] == 0,
        ),
        (
            MeshQuad().refined(2),
            MeshQuad1DG,
            ElementQuad1,
            lambda x: x[0] == 1,
            lambda x: x[0] == 0,
        ),
        (
            MeshQuad().refined(2),
            MeshQuad1DG,
            ElementQuad2,
            lambda x: x[0] == 1,
            lambda x: x[0] == 0,
        ),
        (
            MeshTri().refined(2),
            MeshTri1DG,
            ElementTriP2,
            lambda x: x[0] == 0,
            lambda x: x[0] == 1,
        ),
        (
            MeshTri().refined(2),
            MeshTri1DG,
            ElementTriP2,
            lambda x: x[1] == 0,
            lambda x: x[1] == 1,
        ),
    ]
)
def test_periodic_mesh_assembly(m, mdgtype, etype, check1, check2):
    mp = mdgtype.periodic(m,
                          m.nodes_satisfying(check1),
                          m.nodes_satisfying(check2))

    basis = Basis(mp, etype())
    A = laplace.assemble(basis)
    f = unit_load.assemble(basis)
    D = basis.get_dofs()
    x = solve(*condense(A, f, D=D))

    assert_almost_equal(x.max(), 0.125)
    assert not np.isnan(x).any()


if __name__ == '__main__':
    main()
