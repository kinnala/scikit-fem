import unittest
import pytest

from skfem.assembly import CellBasis
from skfem.mesh import (MeshTri, MeshQuad, MeshTet, MeshLine1, MeshTri2,
                        MeshQuad2)
from skfem.visuals.matplotlib import draw, plot, plot3
from skfem.visuals.svg import draw as drawsvg
from skfem.visuals.svg import plot as plotsvg


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


@pytest.mark.parametrize(
    "mtype",
    [
        MeshTri,
        MeshQuad,
        MeshTri2,
        MeshQuad2,
    ]
)
def test_call_svg_plot(mtype):
    m = mtype()
    svg = drawsvg(m, nrefs=2)
    basis = CellBasis(m, mtype.elem())
    svg_plot = plotsvg(basis, m.p[0], nrefs=2)
