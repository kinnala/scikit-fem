from unittest import TestCase, main

import numpy as np
from numpy.testing import assert_array_equal

from skfem import *


class TestNodality(TestCase):
    """Test for Element.doflocs."""

    elems = [
        ElementLineP1(),
        ElementLineP2(),
        ElementLinePp(1),
        ElementLinePp(3),
        ElementTriP0(),
        ElementTriP1(),
        ElementTriP2(),
        ElementTriMini(),
        ElementQuad0(),
        ElementQuad1(),
        ElementQuad2(),
        ElementQuadS2(),
        ElementQuadP(1),
        ElementQuadP(3),
        ElementTetP0(),
        ElementTetP1(),
        ElementTetP2(),
        ElementTetMini(),
        ElementHex1(),
        ElementHexS2(),
    ]

    def runTest(self):
        for e in self.elems:
            N = e.doflocs.shape[0]
            Ih = np.zeros((N, N))
            for itr in range(N):
                Ih[itr] = e.lbasis(e.doflocs.T, itr)[0]

            # Remove nan-rows: test nodality only on non-nan doflocs.
            #
            # Some elements, such as ElementTriMini might have a combination
            # of nodal dofs and non-nodal dofs.
            #
            # Nodal dof is defined so that there exists a point where the
            # corresponding basis function is one, and other basis functions are
            # zero. Non-nodal dof does not satisfy this property.
            ix = np.isnan(np.sum(Ih, axis=1))
            Nnan = np.sum(ix)
            ixs = np.nonzero(~ix)[0]
            Ih = Ih[ixs].T[ixs].T

            assert_array_equal(Ih, np.eye(N - Nnan),
                               err_msg = "{}".format(type(e)))



class TestNodalityTriRT0(TestCase):

    elem = ElementTriRT0()

    def runTest(self):
        e = self.elem
        N = e.doflocs.shape[0]
        Ih = np.zeros((N, N))
        normals = np.array([[0., -1.],
                            [1 / np.sqrt(2), 1 / np.sqrt(2)],
                            [-1., 0.]]).T
        for itr in range(N):
            # calculate integral of normal component over edge
            Ih[itr] = np.sum(e.lbasis(e.doflocs.T, itr)[0] * normals, axis=0) *\
                np.array([1., np.sqrt(2), 1.])

        assert_array_equal(Ih, np.eye(N),
                       err_msg = "{}".format(type(e)))


class TestComposite(TestCase):

    def runTest(self):
        from skfem.element.element_composite import ElementComposite

        self.check_equivalence(
            ElementComposite(ElementTriP1(),
                             ElementTriP1()),
            ElementVectorH1(ElementTriP1())
        )

    def check_equivalence(self, ec, ev):
        X = np.array([[0.125, 0.1111], [0.0555, 0.6]])
        m = MeshTri.init_refdom()
        mapping = MappingAffine(m)

        for k in range(6):
            for i in [0, 1]:
                # accessing i'th component looks slightly different
                assert_array_equal(
                    ev.gbasis(mapping, X, k)[0].f[i],
                    ec.gbasis(mapping, X, k)[i].f
                )
                for j in [0, 1]:
                    assert_array_equal(
                        ev.gbasis(mapping, X, k)[0].df[i][j],
                        ec.gbasis(mapping, X, k)[i].df[j]
                    )


class TestCompositeMul(TestComposite):

    def runTest(self):

        self.check_equivalence(
            ElementTriP1() * ElementTriP1(),
            ElementVectorH1(ElementTriP1())
        )


class TestCompatibilityWarning(TestCase):

    meshes = [
        MeshTet,
        MeshQuad,
        MeshHex,
        MeshLine,
    ]
    elem = ElementTriP1

    def runTest(self):

        for m in self.meshes:

            def init_incompatible():
                return InteriorBasis(m(), self.elem())

            self.assertRaises(ValueError, init_incompatible)


class TestDerivatives(TestCase):
    """Test values of derivatives."""

    elems = [
        ElementLineP1(),
        ElementLineP2(),
        ElementTriP1(),
        ElementTriP2(),
        ElementTriMini(),
        ElementQuad1(),
        ElementQuad2(),
        ElementQuadS2(),
        ElementTetP1(),
        ElementTetP2(),
        ElementTetMini(),
        ElementHex1(),
        ElementHexS2(),
    ]

    def runTest(self):
        for elem in self.elems:
            eps = 1e-6
            if elem.dim == 1:
                y = np.array([[0.3, 0.3 + eps]])
            elif elem.dim == 2:
                y = np.array([[0.3, 0.3 + eps, 0, 0],
                              [0, 0, 0.3, 0.3 + eps]])
            elif elem.dim == 3:
                y = np.array([[.3, .3 + eps, 0, 0, 0, 0],
                              [0, 0, .3, .3 + eps, 0, 0],
                              [0, 0, 0, 0, .3, .3 + eps]])
            i = 0
            while True:
                try:
                    out = elem.lbasis(y, i)
                except ValueError:
                    break
                diff = (out[0][1] - out[0][0]) / eps
                self.assertAlmostEqual(diff, out[1][0][0], delta=1e-3,
                                       msg='x-derivative for {}th bfun failed for {}'.format(i, elem))
                if elem.dim > 1:
                    diff = (out[0][3] - out[0][2]) / eps
                    self.assertAlmostEqual(diff, out[1][1][3], delta=1e-3,
                                           msg='y-derivative for {}th bfun failed for {}'.format(i, elem))
                if elem.dim == 3:
                    diff = (out[0][5] - out[0][4]) / eps
                    self.assertAlmostEqual(diff, out[1][2][4], delta=1e-3,
                                           msg='z-derivative for {}th bfun failed for {}'.format(i, elem))
                i += 1


if __name__ == '__main__':
    main()
