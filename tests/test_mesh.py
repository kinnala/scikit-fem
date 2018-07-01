import unittest
from skfem.mesh import *

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
            # wrong size inputs (t not matching to Mesh type)
            m = MeshLine(np.array([[0,0],[0,1],[1,0],[1,1]]).T,
                         np.array([[0,1,2]]).T)
        with self.assertRaises(Exception):
            # inputting trasposes
            m = MeshTri(np.array([[0,0],[0,1],[1,0],[1,1]]),
                        np.array([[0,1,2],[1,2,3]]))
