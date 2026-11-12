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


# ---------------------------------------------------------------------------
# skfem.visuals.data
# ---------------------------------------------------------------------------

import json  # noqa: E402

import numpy as np  # noqa: E402

from skfem import (Basis, ElementTriP1, ElementTriP2, ElementTriP0,  # noqa
                   ElementQuad2, ElementTetP2, ElementHex2, ElementVector,
                   ElementLineP2, ElementTriRT0, ElementTriMorley,
                   ElementHex1, ElementTriP1DG)
from skfem.mesh import MeshHex, MeshTri1DG, MeshWedge1  # noqa: E402
from skfem.visuals.data import sample, VisData  # noqa: E402


@pytest.mark.parametrize(
    "mesh,elem",
    [
        (MeshLine1().refined(2), ElementLineP2()),
        (MeshTri().refined(2), ElementTriP2()),
        (MeshQuad().refined(2), ElementQuad2()),
        (MeshTet().refined(1), ElementTetP2()),
        (MeshHex().refined(1), ElementHex2()),
        (MeshTri2.init_circle(1), ElementTriP2()),
        (MeshQuad2().refined(1), ElementQuad2()),
    ]
)
def test_sample_exact(mesh, elem):
    basis = Basis(mesh, elem)
    u = basis.project(lambda x: x[0] ** 2)
    d = sample(basis, {'u': u, 'g': (u, lambda w: w.grad)})
    assert isinstance(d, VisData)
    assert d.meta['nrefs'] == 1
    assert d.npoints == d.nelements * d.nloc
    assert d.ncells == d.nelements * d.nsub
    assert d.point_data['g'].shape == (d.npoints, mesh.dim())
    if not isinstance(mesh, (MeshTri2, MeshQuad2)):  # curved: projection
        np.testing.assert_allclose(d.point_data['u'], d.points[:, 0] ** 2,
                                   atol=1e-12)
        np.testing.assert_allclose(d.point_data['g'][:, 0],
                                   2 * d.points[:, 0], atol=1e-10)
    # coincident points are welded
    assert d.weld.max() + 1 < d.npoints or mesh.nelements == 1
    payload = json.loads(d.to_json())
    assert payload['nel'] == d.nelements
    assert [f['name'] for f in payload['fields']] == ['u', 'g']


def test_sample_curved_geometry():
    m = MeshTri2.init_circle(1)
    d = sample(m, nrefs=2)
    # all points of curved elements are inside the unit disk
    assert np.max(np.linalg.norm(d.points, axis=1)) <= 1 + 1e-12
    # some subdivision points lie on the circle (not only the vertices)
    on_circle = np.abs(np.linalg.norm(d.points, axis=1) - 1) < 1e-12
    assert np.sum(on_circle) > 2 * np.sum(
        np.abs(np.linalg.norm(m.p[:, :m.nvertices], axis=0) - 1) < 1e-12)


def test_sample_discontinuous():
    m = MeshTri().refined(2)
    basis = Basis(m, ElementTriP0())
    y = np.arange(m.nelements, dtype=np.float64)
    d = sample(basis, y)
    assert d.meta['nrefs'] == 0
    np.testing.assert_array_equal(d.point_data['u'],
                                  np.repeat(y, d.nloc))
    # also for elementwise values given on the mesh
    d2 = sample(m, y)
    np.testing.assert_array_equal(d2.point_data['u'], d.point_data['u'])


def test_sample_mesh_nodal():
    m = MeshTri().refined(2)
    d = sample(m, m.p[1], nrefs=2)
    np.testing.assert_allclose(d.point_data['u'], d.points[:, 1],
                               atol=1e-12)
    assert sample(m).fields == []


def test_sample_vector_composite():
    m = MeshTri().refined(2)
    b = Basis(m, ElementVector(ElementTriP2()) * ElementTriP1())
    d = sample(b, b.zeros())
    assert d.fields == ['u0', 'u1']
    assert d.field_shapes == {'u0': (2,), 'u1': ()}
    b1 = Basis(m, ElementVector(ElementTriP1()))
    b2 = Basis(m, ElementTriP1())
    d = sample(b1 * b2, (b1 * b2).zeros())
    assert d.fields == ['u0', 'u1']
    d = sample(b1, {'v': b1.zeros(), 'p': (b2, b2.ones())})
    np.testing.assert_allclose(d.point_data['p'], 1.)
    d = sample(Basis(m, ElementTriRT0()), Basis(m, ElementTriRT0()).zeros())
    assert d.field_shapes['u'] == (2,)
    d = sample(Basis(m, ElementTriMorley()),
               Basis(m, ElementTriMorley()).zeros())
    assert d.meta['nrefs'] == 1


def test_sample_subset_and_wedge():
    m = MeshTri().refined(2)
    d = sample(Basis(m, ElementTriP1(), elements=[0, 1, 2]), m.p[0])
    assert d.nelements == 3
    np.testing.assert_array_equal(np.unique(d.element), [0, 1, 2])
    d = sample(m, elements=[4, 5])
    assert d.nelements == 2
    d = sample(MeshWedge1(), MeshWedge1().p[0])
    assert d.cell_type == 'wedge'


def test_sample_dg_mesh():
    m = MeshTri1DG.init_tensor(np.linspace(0, 1, 4), np.linspace(0, 1, 4),
                               periodic=[0])
    d = sample(Basis(m, ElementTriP1DG()), m.p[0])
    assert d.points.shape[0] == 3 * m.nelements


def test_visdata_conversions(tmp_path):
    m = MeshHex().refined(1)
    b = Basis(m, ElementVector(ElementHex1()))
    d = sample(b, b.zeros())
    tris = d.boundary_triangles()
    # 6 sides x 4 quads x 2 triangles
    assert tris.shape == (48, 3)
    mesh = d.to_mesh()
    assert mesh.nelements == d.ncells
    d.save(str(tmp_path / 'out.vtk'))
    d2 = sample(MeshQuad().refined(1), MeshQuad().refined(1).p[0])
    assert d2.triangles().shape == (2 * d2.ncells, 3)


def test_refinterp_backwards_compatible():
    m = MeshTri().refined(2)
    b = Basis(m, ElementVector(ElementTriP2()))
    y = b.project(lambda x: np.array([x[0], 2 * x[1]]))
    M, Y = b.refinterp(y, nrefs=2)
    assert isinstance(M, MeshTri)
    assert len(Y) == 2 * M.nvertices
    np.testing.assert_allclose(Y[:M.nvertices], M.p[0], atol=1e-12)
    np.testing.assert_allclose(Y[M.nvertices:], 2 * M.p[1], atol=1e-12)
    M, Y = Basis(MeshTri2.init_circle(), ElementTriP2()).refinterp(
        MeshTri2.init_circle().p[0], nrefs=1)
    assert isinstance(M, MeshTri)


def test_default_backend():
    from skfem.visuals import set_default_backend, get_default_backend
    assert get_default_backend() == 'matplotlib'
    try:
        set_default_backend('svg')
        assert MeshTri().draw().svg.startswith('<svg')
    finally:
        set_default_backend('matplotlib')
    with pytest.raises(ModuleNotFoundError):
        set_default_backend('nonexistent')
