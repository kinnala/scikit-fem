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


if __name__ == '__main__':
    unittest.main()
