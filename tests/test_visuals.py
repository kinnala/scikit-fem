import unittest

from skfem.mesh import MeshTri, MeshQuad, MeshTet, MeshLine1
from skfem.visuals.matplotlib import draw, plot, plot3


class CallDraw(unittest.TestCase):
    mesh_type = MeshTri

    def runTest(self):
        m = self.mesh_type()
        draw(m)


class CallDrawQuad(CallDraw):
    mesh_type = MeshQuad


class CallDrawTet(CallDraw):
    mesh_type = MeshTet


class CallPlot(unittest.TestCase):
    mesh_type = MeshTri

    def runTest(self):
        m = self.mesh_type()
        plot(m, m.p[0])


class CallPlotQuad(CallPlot):
    mesh_type = MeshQuad


class CallPlotLine(CallPlot):
    mesh_type = MeshLine1


class CallPlot3(unittest.TestCase):
    mesh_type = MeshTri

    def runTest(self):
        m = self.mesh_type()
        plot3(m, m.p[0])
