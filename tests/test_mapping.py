import unittest
import numpy as np

from skfem.mesh import MeshHex, MeshQuad, MeshTri, MeshTet
from skfem.element import ElementHex1, ElementQuad1, ElementHex2
from skfem.assembly import FacetBasis
from skfem.mapping import MappingAffine


class TestIsoparamNormals(unittest.TestCase):
    """Test that normals on x[i] == 0 are correct."""

    mesh = MeshHex
    elem = ElementHex1

    def runTest(self):
        m = self.mesh().refined()
        e = self.elem()
        fb = FacetBasis(m, e)
        x = fb.global_coordinates().value
        eps = 1e-6
        for itr in range(m.p.shape[0]):
            case = (x[itr] < eps) * (x[itr] > -eps)
            for jtr in range(m.p.shape[0]):
                normals = fb.normals.value[jtr][case]
                if itr == jtr:
                    self.assertTrue((normals == -1).all())
                else:
                    self.assertTrue((np.abs(normals) < 1e-14).all())


class TestIsoparamNormalsQuad(TestIsoparamNormals):

    mesh = MeshQuad
    elem = ElementQuad1


class TestIsoparamNormalsHex2(TestIsoparamNormals):

    elem = ElementHex2


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
        return ((np.abs(y) < 1. + 1e-12).all()
                and (np.abs(y) > 0. - 1e-12).all())

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


class TestInverseMappingHex2(TestInverseMappingHex):
    """This should be equivalent to TestInverseMappingHex."""

    element = ElementHex2


def test_mapping_memory_optimization():

    m = MeshTet.init_tensor(np.linspace(0, 1, 100),
                            np.linspace(0, 1, 100),
                            np.linspace(0, 1, 100))
    m = m.with_subdomains({
        'omega0': [0, 1, 2, 3, 4, 5],
    })
    orig = MappingAffine(m)
    opt = MappingAffine(m, tind=m.subdomains['omega0'])
    assert len(orig.detA) > len(opt.detA)
