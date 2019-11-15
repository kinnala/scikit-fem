import unittest

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
                    self.assertTrue((fb.normals[jtr][case] == 0).all())


class TestIsoparamNormalsQuad(TestIsoparamNormals):
    mesh = MeshQuad
    elem = ElementQuad1


class TestNoCrash(unittest.TestCase):
    """Test that mappings don't crash for non-rectangular elements."""

    def runTest(self):
        m0 = MeshQuad()
        m = MeshQuad([[0, 1, 1, 0],
                      [0, .1, 1, 1]],
                     m0.t)
        e = ElementQuad1()
        FacetBasis(m0, e)
        FacetBasis(m, e)


if __name__ == '__main__':
    unittest.main()
