from unittest import TestCase
import pytest

import matplotlib.pyplot as plt

from skfem.assembly import CellBasis
from skfem.mesh import (MeshTri, MeshQuad, MeshTet, MeshLine1, MeshTri2,
                        MeshQuad2)
from skfem.visuals.matplotlib import draw, plot, plot3
from skfem.visuals.svg import draw as drawsvg
from skfem.visuals.svg import plot as plotsvg


class CallDraw(TestCase):
    mesh_type = MeshTri

    def runTest(self):
        m = self.mesh_type()
        draw(m,
             aspect=1.1,
             facet_numbering=True,
             node_numbering=True,
             element_numbering=True)


class CallDrawQuad(CallDraw):
    mesh_type = MeshQuad


class CallDrawTet(CallDraw):
    mesh_type = MeshTet


class CallPlot(TestCase):
    mesh_type = MeshTri

    def runTest(self):
        m = self.mesh_type()
        plot(m, m.p[0])


class CallPlotQuad(CallPlot):
    mesh_type = MeshQuad


class CallPlotLine(CallPlot):
    mesh_type = MeshLine1


class CallPlot3(TestCase):
    mesh_type = MeshTri

    def runTest(self):
        m = self.mesh_type()
        plot3(m, m.p[0])


class CallPlotBasis(TestCase):
    mesh_type = MeshTri

    def runTest(self):
        m = self.mesh_type()
        basis = CellBasis(m, self.mesh_type.elem())
        y = basis.project(lambda x: x[0])
        plot(basis,
             y,
             nrefs=1,
             figsize=(4, 4),
             aspect=1.1,
             cmap=plt.get_cmap('viridis'),
             shading='gouraud',
             vmin=0,
             vmax=1,
             levels=3,
             colorbar=True)


class CallPlotBasisQuad(CallPlotBasis):
    mesh_type = MeshQuad


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
