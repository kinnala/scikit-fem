import unittest


from skfem.mesh import *
from skfem.visuals.matplotlib import *


class CallDraw(unittest.TestCase):
    mesh_type = MeshTri

    def runTest(self):
        # Mesh.remove_elements
        m = self.mesh_type()
        draw(m)


class CallDrawQuad(CallDraw):
    mesh_type = MeshQuad


class CallDrawTet(CallDraw):
    mesh_type = MeshTet
