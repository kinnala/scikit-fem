import unittest
import numpy as np

from skfem import *


class TestIsoparamNormals(unittest.TestCase):
    """Test that normals on x[i] == 0 are correct."""
    mesh = MeshHex
    elem = ElementHex1

    def runTest(self):
        m = self.mesh()
        m.refine()
        e = self.elem()
        fb = FacetBasis(m, e)
        x = fb.global_coordinates()
        eps = 1e-6
        for itr in range(m.p.shape[0]):
            case = (x[itr] < eps) * (x[itr] > -eps)
            for jtr in range(m.p.shape[0]):
                if itr == jtr:
                    self.assertTrue((fb.normals[jtr][case] == -1).all())
                else:
                    self.assertTrue((np.abs(fb.normals[jtr][case]) < 1e-14).all())


class TestIsoparamNormalsQuad(TestIsoparamNormals):
    mesh = MeshQuad
    elem = ElementQuad1


class TestInverseMapping(unittest.TestCase):
    """Test that inverse mapping works for non-rectangular elements."""

    element = ElementQuad1

    def initialize_meshes(self):
        m0 = MeshQuad()
        m = MeshQuad([[0, 1, 1, 0],
                      [0, .9, 1, 1]],
                     m0.t)
        return m

    def within_refelem(self, y):
        return (np.abs(y) < 1.0 + 1e-12).all()

    def runTest(self):
        m = self.initialize_meshes()
        e = self.element()
        fb = FacetBasis(m, e)

        x = fb.mapping.G(fb.X, find=fb.find)
        Y0 = fb.mapping.invF(x, tind=fb.mesh.f2t[0, fb.find])

        assert self.within_refelem(Y0)


class TestInverseMappingHex(TestInverseMapping):

    element = ElementHex1

    def initialize_meshes(self):
        m0 = MeshHex()
        m = MeshHex(np.array([[0., 0., 0.],
                              [0., 0., 1.],
                              [0., 1., 0.],
                              [1., 0.7, 0.7],
                              [0., 1., 1.],
                              [1., 0., 1.],
                              [1., 1., 0.],
                              [1., 1., 1.]]).T, m0.t)
        return m


class TestMortarPair(unittest.TestCase):
    """Check that mapped points match."""

    mesh1_type = MeshTri
    mesh2_type = MeshTri
    nrefs1 = 2
    nrefs2 = 3
    translate_y = 0.0

    def init_meshes(self):
        m1 = self.mesh1_type()
        m1.refine(self.nrefs1)
        m2 = self.mesh2_type()
        m2.refine(self.nrefs2)
        m2.translate((1.0, self.translate_y))
        return m1, m2

    def runTest(self):
        m1, m2 = self.init_meshes()
        mp = MortarPair.init_2D(m1, m2,
                                m1.facets_satisfying(lambda x: x[0] == 1.),
                                m2.facets_satisfying(lambda x: x[0] == 1.))
        test_points = np.array([np.linspace(0., 1., 7)])
        self.assertTrue((mp.mapping1.G(test_points) -
                         mp.mapping1.G(test_points) < 1e-10).all())


class TestMortarPairTriQuad(TestMortarPair):
    mesh1_type = MeshTri
    mesh2_type = MeshQuad


class TestMortarPairQuadQuad(TestMortarPair):
    mesh1_type = MeshQuad
    mesh2_type = MeshQuad


class TestMortarPairNoMatch1(TestMortarPair):
    mesh1_type = MeshQuad
    mesh2_type = MeshTri
    translate_y = 0.1


class TestMortarPairNoMatch2(TestMortarPair):
    mesh1_type = MeshQuad
    mesh2_type = MeshTri
    translate_y = -np.pi / 10.


if __name__ == '__main__':
    unittest.main()
