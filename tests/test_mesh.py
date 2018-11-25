import unittest
import numpy as np
from skfem.mesh import *


class MeshTests(unittest.TestCase):
    """Test some of the methods in mesh classes
    that are not tested elsewhere."""
    def runTest(self):
        # Mesh.remove_elements
        m = MeshTri()
        m.refine()
        M = m.remove_elements([0])
        self.assertEqual(M.t.shape[1], 7)

        # Mesh.scale, Mesh.translate
        m = MeshHex()
        m.scale(0.5)
        m.translate((0.5, 0.5, 0.5))
        self.assertGreater(np.min(m.p), 0.4999)

        # Mesh3D.facets_satisfying
        self.assertEqual(len(m.facets_satisfying(lambda x,y,z: x==0.5)), 1)

class FaultyInputs(unittest.TestCase):
    """Check that faulty meshes are detected by the constructors."""
    def runTest(self):
        with self.assertRaises(Exception):
            # point belonging to no element
            m = MeshTri(np.array([[0,0],[0,1],[1,0],[1,1]]).T,
                        np.array([[0,1,2]]).T)
        with self.assertRaises(Exception):
            # wrong size inputs (t not matching to Mesh type)
            m = MeshTet(np.array([[0,0],[0,1],[1,0],[1,1]]).T,
                        np.array([[0,1,2]]).T)
        with self.assertRaises(Exception):
            # inputting trasposes
            m = MeshTri(np.array([[0,0],[0,1],[1,0],[1,1]]),
                        np.array([[0,1,2],[1,2,3]]))
        with self.assertRaises(Exception):
            # floats in element connectivity
            m = MeshTri(np.array([[0,0],[0,1],[1,0],[1,1]]).T,
                        np.array([[0.0,1.0,2.0],[1.0,2.0,3.0]]).T)

class Loading(unittest.TestCase):
    """Check that Mesh.load works properly."""
    def runTest(self):
        # submeshes
        m = MeshTet.load('docs/examples/box.msh')
        #self.assertTrue((m.boundaries['front'].p == m.submesh(lambda x,y,z: z==1).p).all())
        #self.assertTrue((m.boundaries['back'].p == m.submesh(lambda x,y,z: z==0).p).all())
        #self.assertTrue((m.boundaries['top'].p == m.submesh(lambda x,y,z: y==1).p).all())
        self.assertTrue((m.boundaries['top'] == m.facets_satisfying(lambda x,y,z: y==1)).all())
        self.assertTrue((m.boundaries['back'] == m.facets_satisfying(lambda x,y,z: z==0)).all())
        self.assertTrue((m.boundaries['front'] == m.facets_satisfying(lambda x,y,z: z==1)).all())
        #self.assertTrue((m.boundaries['top'].edges == m.submesh(lambda x,y,z: y==1).edges).all())
        #self.assertTrue((m.boundaries['back'].edges == m.submesh(lambda x,y,z: z==0).edges).all())
        #self.assertTrue((m.boundaries['front'].edges == m.submesh(lambda x,y,z: z==1).edges).all())
        m = MeshTri.load('docs/examples/square.msh')
        self.assertTrue((m.boundaries['top'] == m.facets_satisfying(lambda x,y: y==1)).all())
        self.assertTrue((m.boundaries['left'] == m.facets_satisfying(lambda x,y: x==0)).all())
        self.assertTrue((m.boundaries['right'] == m.facets_satisfying(lambda x,y: x==1)).all())


class RefinePreserveSubsets(unittest.TestCase):
    """Check that uniform refinement preserves named boundaries."""
    def runTest(self):
        m = MeshTri()
        m.refine(3)
        m.boundaries = {'test': m.facets_satisfying(lambda x,y: x==0.0)}

        m.refine()

        self.assertTrue((np.sort(m.boundaries['test'])
                         == np.sort(m.facets_satisfying(lambda x,y: x==0.0))).all())


