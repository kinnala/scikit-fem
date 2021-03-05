import unittest
from pathlib import Path

import numpy as np

from skfem.mesh import Mesh, MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri


class MeshTests(unittest.TestCase):
    """Test some of the methods in mesh classes
    that are not tested elsewhere."""

    def runTest(self):
        # Mesh.remove
        m = MeshTri().refined()
        M = m.remove_elements(np.array([0]))
        self.assertEqual(M.t.shape[1], 7)

        # Mesh.define_boundary
        m.define_boundary('foo', lambda x: x[0] == 0.)
        self.assertEqual(m.boundaries['foo'].size, 2)

        # Mesh.define_boundary (internal)
        m.define_boundary('bar', lambda x: x[0] == 1./2, boundaries_only=False)
        self.assertEqual(m.boundaries['bar'].size, 2)

        m = MeshHex().scaled(0.5).translated((0.5, 0.5, 0.5))
        self.assertGreater(np.min(m.p), 0.4999)

        # Mesh3D.facets_satisfying
        self.assertEqual(len(m.facets_satisfying(lambda x: x[0] == 0.5)), 1)


class FaultyInputs(unittest.TestCase):
    """Check that faulty meshes are detected by the constructors."""

    def _runTest(self):  # disabled
        with self.assertRaises(Exception):
            # point belonging to no element
            MeshTri(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T,
                    np.array([[0, 1, 2]]).T)
        with self.assertRaises(Exception):
            # wrong size inputs (t not matching to Mesh type)
            MeshTet(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T,
                    np.array([[0, 1, 2]]).T)
        with self.assertRaises(Exception):
            # inputting trasposes
            MeshTri(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]),
                    np.array([[0, 1, 2], [1, 2, 3]]))
        with self.assertRaises(Exception):
            # floats in element connectivity
            MeshTri(np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T,
                    np.array([[0.0, 1.0, 2.0], [1.0, 2.0, 3.0]]).T)


class Loading(unittest.TestCase):
    """Check that Mesh.load works properly."""

    def runTest(self):
        # submeshes
        path = Path(__file__).parents[1] / 'docs' / 'examples' / 'meshes'
        m = MeshTet.load(str(path / 'box.msh'))
        self.assertTrue((m.boundaries['top']
                         == m.facets_satisfying(lambda x: x[1] == 1)).all())
        self.assertTrue((m.boundaries['back']
                         == m.facets_satisfying(lambda x: x[2] == 0)).all())
        self.assertTrue((m.boundaries['front']
                         == m.facets_satisfying(lambda x: x[2] == 1)).all())
        m = MeshTri.load(str(path / 'square.msh'))
        self.assertTrue((m.boundaries['top']
                         == m.facets_satisfying(lambda x: x[1] == 1)).all())
        self.assertTrue((m.boundaries['left']
                         == m.facets_satisfying(lambda x: x[0] == 0)).all())
        self.assertTrue((m.boundaries['right']
                         == m.facets_satisfying(lambda x: x[0] == 1)).all())


class RefinePreserveSubsets(unittest.TestCase):
    """Check that uniform refinement preserves named boundaries."""

    def _runTest(self):  # disabled
        for mtype in (MeshLine, MeshTri, MeshQuad):
            m = mtype().refined(2)
            boundaries = [('external', lambda x: x[0] * (1. - x[0]) == 0.0),
                          ('internal', lambda x: x[0] == 0.5)]
            for name, handle in boundaries:
                m.define_boundary(name, handle, boundaries_only=False)
            m = m.refined()
            for name, handle in boundaries:
                A = np.sort(m.boundaries[name])
                B = np.sort(m.facets_satisfying(handle))
                self.assertTrue((A == B).all())
            m = m.refined(2)
            for name, handle in boundaries:
                A = np.sort(m.boundaries[name])
                B = np.sort(m.facets_satisfying(handle))
                self.assertTrue((A == B).all())


class SaveLoadCycle(unittest.TestCase):
    """Save to temporary file and check import/export cycles."""
    cls = MeshTet

    def runTest(self):
        from tempfile import NamedTemporaryFile
        m = self.cls().refined(2)
        f = NamedTemporaryFile(delete=False)
        m.save(f.name + ".vtk")
        with self.assertWarnsRegex(UserWarning, '^Unable to load tagged'):
            m2 = Mesh.load(f.name + ".vtk")
        self.assertTrue(((m.p[0, :] - m2.p[0, :]) < 1e-6).all())


class SaveLoadCycleHex(SaveLoadCycle):

    cls = MeshHex


class SerializeUnserializeCycle(unittest.TestCase):
    """Check to_dict/initialize cycles."""

    clss = [MeshTet,
            MeshTri,
            MeshHex,
            MeshQuad]

    def runTest(self):
        for cls in self.clss:
            m = cls().refined(2)
            m.define_boundary('down', lambda x: x[0] == 0)
            m.define_subdomain('up', lambda x: x[0] > 0.5)
            M = cls.from_dict(m.to_dict())
            self.assertTrue(np.sum(m.p - M.p) < 1e-13)
            self.assertTrue(np.sum(m.t - M.t) < 1e-13)
            for k in m.boundaries:
                self.assertTrue((m.boundaries[k] == M.boundaries[k]).all())
            for k in m.subdomains:
                self.assertTrue((m.subdomains[k] == M.subdomains[k]).all())


class TestBoundaryEdges(unittest.TestCase):

    def runTest(self):
        m = MeshTet()
        # default mesh has all edges on the boundary
        self.assertEqual(len(m.boundary_edges()), m.edges.shape[1])
        # check that there is a correct amount of boundary edges:
        # 12 (cube edges) * 2 (per cube edge)
        # + 6 (cube faces) * 8 (per cube face)
        # = 72 edges
        self.assertTrue(len(m.refined().boundary_edges()) == 72)


class TestBoundaryEdges2(unittest.TestCase):

    def runTest(self):
        m = MeshHex()
        # default mesh has all edges on the boundary
        self.assertTrue(len(m.boundary_edges()) == m.edges.shape[1])
        # check that there is a correct amount of boundary edges:
        # 12 (cube edges) * 2 (per cube edge)
        # + 6 (cube faces) * 4 (per cube face)
        # = 48 edges
        self.assertEqual(len(m.refined().boundary_edges()), 48)


class TestMeshAddition(unittest.TestCase):

    def runTest(self):
        m = MeshTri()
        M = MeshTri().translated((1.0, 0.0))
        mesh = m + M
        self.assertTrue(mesh.p.shape[1] == 6)
        self.assertTrue(mesh.t.shape[1] == 4)


class TestMeshQuadSplit(unittest.TestCase):

    def runTest(self):
        from docs.examples.ex17 import mesh
        mesh_tri = mesh.to_meshtri()

        for s in mesh.subdomains:
            self.assertEqual(np.setdiff1d(*[m.t.T[m.subdomains[s]]
                                            for m in [mesh, mesh_tri]]).size,
                             0)

        for b in mesh.boundaries:
            np.testing.assert_array_equal(*[m.facets.T[m.boundaries[b]]
                                            for m in [mesh, mesh_tri]])

    def runRefineTest(self):
        mesh = MeshQuad()
        mesh.define_boundary('left', lambda x: x[0] == 0)
        mesh = mesh.refined()
        mesh_tri = mesh.to_meshtri()

        for b in mesh.boundaries:
            np.testing.assert_array_equal(*[m.facets.T[m.boundaries[b]]
                                            for m in [mesh, mesh_tri]])


class TestAdaptiveSplitting1D(unittest.TestCase):

    def runTest(self):

        m = MeshLine()

        for itr in range(10):
            prev_t_size = m.t.shape[1]
            prev_p_size = m.p.shape[1]
            m = m.refined([prev_t_size - 1])

            # check that new size is current size + 1
            self.assertEqual(prev_t_size, m.t.shape[1] - 1)
            self.assertEqual(prev_p_size, m.p.shape[1] - 1)


class TestAdaptiveSplitting2D(unittest.TestCase):

    def runTest(self):

        m = MeshTri()
        prev_t_size = -1

        for itr in range(5):
            red_ix = prev_t_size - 1 if prev_t_size != -1\
                else m.t.shape[1] - 1
            prev_t_size = m.t.shape[1]
            prev_p_size = m.p.shape[1]
            m = m.refined([red_ix])

            # check that new size is current size + 4
            self.assertEqual(prev_t_size, m.t.shape[1] - 4)
            self.assertEqual(prev_p_size, m.p.shape[1] - 3)


if __name__ == '__main__':
    unittest.main()
