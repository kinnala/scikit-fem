from unittest import TestCase
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay
from numpy.testing import assert_array_equal, assert_almost_equal

from skfem.mesh import (Mesh, MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri,
                        MeshTri2, MeshQuad2, MeshTet2, MeshHex2, MeshLine1DG,
                        MeshQuad1DG, MeshHex2, MeshTri1DG)
from skfem.assembly import Basis, LinearForm
from skfem.element import ElementTetP1
from skfem.utils import projection
from skfem.io.meshio import to_meshio, from_meshio
from skfem.io.json import to_dict, from_dict


MESH_PATH = Path(__file__).parents[1] / 'docs' / 'examples' / 'meshes'


class MeshTests(TestCase):
    """Test some of the methods in mesh classes
    that are not tested elsewhere."""

    def runTest(self):
        # Mesh.remove
        m = MeshTri().refined()
        M = m.remove_elements(np.array([0]))
        self.assertEqual(M.t.shape[1], 7)

        # boundaries
        M = m.with_boundaries({
            'foo': lambda x: x[0] == 0.,
        })
        self.assertEqual(M.boundaries['foo'].size, 2)

        m = MeshHex().scaled(0.5).translated((0.5, 0.5, 0.5))
        self.assertGreater(np.min(m.p), 0.4999)

        # Mesh3D.facets_satisfying
        self.assertEqual(len(m.facets_satisfying(lambda x: x[0] == 0.5)), 1)

        # test restrict
        m = MeshQuad().refined().with_subdomains({
            'left': lambda x: x[0] < 0.5,
            'right': lambda x: x[0] > 0.5,
        })
        assert_array_equal(
            m.remove_elements(m.normalize_elements('left')).p,
            m.restrict('right').p
        )
        assert_array_equal(
            m.remove_elements(m.normalize_elements('left')).t,
            m.restrict('right').t
        )


class FaultyInputs(TestCase):
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


class Loading(TestCase):
    """Check that Mesh.load works properly."""

    def runTest(self):
        # submeshes
        m = MeshTet.load(MESH_PATH / 'box.msh')
        self.assertTrue((m.boundaries['top']
                         == m.facets_satisfying(lambda x: x[1] == 1)).all())
        self.assertTrue((m.boundaries['back']
                         == m.facets_satisfying(lambda x: x[2] == 0)).all())
        self.assertTrue((m.boundaries['front']
                         == m.facets_satisfying(lambda x: x[2] == 1)).all())
        m = MeshTri.load(MESH_PATH / 'square.msh')
        self.assertTrue((m.boundaries['top']
                         == m.facets_satisfying(lambda x: x[1] == 1)).all())
        self.assertTrue((m.boundaries['left']
                         == m.facets_satisfying(lambda x: x[0] == 0)).all())
        self.assertTrue((m.boundaries['right']
                         == m.facets_satisfying(lambda x: x[0] == 1)).all())


class SerializeUnserializeCycle(TestCase):
    """Check to_dict/initialize cycles."""

    clss = [MeshTet,
            MeshTri,
            MeshHex,
            MeshQuad]

    def runTest(self):
        for cls in self.clss:
            m = (cls()
                 .refined(2)
                 .with_boundaries({'down': lambda x: x[0] == 0,})
                 .with_subdomains({'up': lambda x: x[0] > 0.5}))
            M = from_dict(cls, to_dict(m))
            self.assertTrue(np.sum(m.p - M.p) < 1e-13)
            self.assertTrue(np.sum(m.t - M.t) < 1e-13)
            for k in m.boundaries:
                self.assertTrue((m.boundaries[k] == M.boundaries[k]).all())
            for k in m.subdomains:
                self.assertTrue((m.subdomains[k] == M.subdomains[k]).all())


class TestBoundaryEdges(TestCase):

    def runTest(self):
        m = MeshTet()
        # default mesh has all edges on the boundary
        self.assertEqual(len(m.boundary_edges()), m.edges.shape[1])
        # check that there is a correct amount of boundary edges:
        # 12 (cube edges) * 2 (per cube edge)
        # + 6 (cube faces) * 8 (per cube face)
        # = 72 edges
        self.assertTrue(len(m.refined().boundary_edges()) == 72)


class TestBoundaryEdges2(TestCase):

    def runTest(self):
        m = MeshHex()
        # default mesh has all edges on the boundary
        self.assertTrue(len(m.boundary_edges()) == m.edges.shape[1])
        # check that there is a correct amount of boundary edges:
        # 12 (cube edges) * 2 (per cube edge)
        # + 6 (cube faces) * 4 (per cube face)
        # = 48 edges
        self.assertEqual(len(m.refined().boundary_edges()), 48)


class TestMeshAddition(TestCase):

    def runTest(self):
        m = MeshTri()
        M = MeshTri().translated((1.0, 0.0))
        mesh = m + M
        self.assertTrue(mesh.p.shape[1] == 6)
        self.assertTrue(mesh.t.shape[1] == 4)


class TestMeshQuadSplit(TestCase):

    def runTest(self):
        from docs.examples.ex17 import mesh
        for style in [None, 'x']:
            mesh_tri = mesh.to_meshtri(style=style)

            for s in mesh.subdomains:
                self.assertEqual(
                    np.setdiff1d(*[m.t.T[m.subdomains[s]]
                                   for m in [mesh, mesh_tri]]).size, 0)

            for b in mesh.boundaries:
                np.testing.assert_array_equal(*[m.facets.T[m.boundaries[b]]
                                                for m in [mesh, mesh_tri]])

        for style in [None, 'x']:
            mesh = MeshQuad().refined().with_boundaries({
                'left': lambda x: x[0] == 0,
            })
            mesh_tri = mesh.to_meshtri(style=style)

            for b in mesh.boundaries:
                np.testing.assert_array_equal(*[m.facets.T[m.boundaries[b]]
                                                for m in [mesh, mesh_tri]])


class TestAdaptiveSplitting1D(TestCase):

    def runTest(self):

        m = MeshLine()

        for itr in range(10):
            prev_t_size = m.t.shape[1]
            prev_p_size = m.p.shape[1]
            m = m.refined([prev_t_size - 1])

            # check that new size is current size + 1
            self.assertEqual(prev_t_size, m.t.shape[1] - 1)
            self.assertEqual(prev_p_size, m.p.shape[1] - 1)


class TestAdaptiveSplitting2D(TestCase):

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


def test_adaptive_splitting_3d():
    m = MeshTet()
    for itr in range(10):
        M = m.refined([itr, itr + 1, itr + 2])
        assert M.is_valid()
        m = M

def test_adaptive_splitting_3d_0():
    m = MeshTet()
    for itr in range(10):
        m = m.refined([itr, itr + 1])
        assert m.is_valid()

def test_adaptive_splitting_3d_1():
    m = MeshTet()
    for itr in range(50):
        m = m.refined([itr])
        assert m.is_valid()

def test_adaptive_splitting_3d_2():
    m = MeshTet()
    for itr in range(5):
        m = m.refined(np.arange(m.nelements, dtype=np.int64))
        assert m.is_valid()

def test_adaptive_splitting_3d_3():
    # adaptively refine one face of a cube, check that the mesh parameter h
    # is approximately linear w.r.t to distance from the face
    m = MeshTet.init_tensor(np.linspace(0, 1, 3),
                            np.linspace(0, 1, 3),
                            np.linspace(0, 1, 3))

    for itr in range(15):
        m = m.refined(m.f2t[0, m.facets_satisfying(lambda x: x[0] == 0)])

    @LinearForm
    def hproj(v, w):
        return w.h * v

    basis = Basis(m, ElementTetP1())
    h = projection(hproj, basis)

    funh = basis.interpolator(h)

    xs = np.vstack((
        np.linspace(0, .5, 20),
        np.zeros(20) + .5,
        np.zeros(20) + .5,
    ))
    hs = funh(xs)

    assert np.max(np.abs(hs - xs[0])) < 0.063


def test_adaptive_splitting_3d_4():
    # check that the same mesh is reproduced by any future versions
    m = MeshTet.init_tensor(np.linspace(0, 1, 2),
                            np.linspace(0, 1, 2),
                            np.linspace(0, 1, 2))

    m = m.refined(m.f2t[0, m.facets_satisfying(lambda x: x[0] == 0)])

    assert_array_equal(
        m.p,
        np.array([[0. , 0. , 1. , 1. , 0. , 0. , 1. , 1. , 0.5],
                  [0. , 1. , 0. , 1. , 0. , 1. , 0. , 1. , 0.5],
                  [0. , 0. , 0. , 0. , 1. , 1. , 1. , 1. , 0.5]])
    )

    assert_array_equal(
        m.t,
        np.array([[5, 3, 3, 5, 6, 6, 1, 4, 1, 2, 2, 4],
                  [0, 0, 0, 0, 0, 0, 7, 7, 7, 7, 7, 7],
                  [1, 1, 2, 4, 2, 4, 5, 5, 3, 3, 6, 6],
                  [8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8]])
    )


def test_adaptive_splitting_3d_5():
    # random refine
    m = MeshTet()

    np.random.seed(1337)
    for itr in range(10):
        m = m.refined(
            np.unique(
                np.random.randint(0,
                                  m.t.shape[1],
                                  size=int(0.3 * m.t.shape[1]))))
        assert m.is_valid()


@pytest.mark.parametrize(
    "m,seed",
    [
        (MeshTet(), 0),
        (MeshTet(), 1),  # problems
        (MeshTet(), 2),
        (MeshTet(), 3),
        (MeshTet().refined(), 10),
    ]
)
def test_adaptive_random_splitting(m, seed):

    np.random.seed(seed)
    points = np.hstack((m.p, np.random.rand(m.p.shape[0], 100)))
    tri = Delaunay(points.T)
    m = type(m)(points, tri.simplices.T)
    assert m.is_valid()

    for itr in range(3):
        M = m.refined(np.unique(
            np.random.randint(0,
                              m.t.shape[1],
                              size=int(0.3 * m.t.shape[1]))))
        assert M.is_valid()

        # test that all boundary facets are on the boundary
        bfacets = M.facets[:, M.boundary_facets()]
        bmidp = np.array([np.mean(M.p[itr, bfacets], axis=0)
                          for itr in range(3)])
        assert (np.isclose(bmidp, 0) + np.isclose(bmidp, 1)).any(axis=0).all()

        m = M


class TestMirrored(TestCase):

    def runTest(self):

        m1 = MeshTet()
        m2 = m1.mirrored((1, 0, 0))
        m3 = m1.mirrored((0, 1, 0))
        m4 = m1.mirrored((0, 0, 1))
        m = m1 + m2 + m3 + m4

        self.assertEqual(m.nvertices, 20)
        self.assertEqual(m.nelements, 20)

        m = MeshTri.init_tensor(
            np.linspace(1,2,2),
            np.linspace(1,2,2),
        )
        m = m + m.mirrored((0, 1), (2, 1))

        self.assertEqual(len(m.boundary_facets()), 6)
        self.assertEqual(m.nvertices, 6)


class TestFinder1DRefined(TestCase):

    def runTest(self):

        for itr in range(5):
            finder = MeshLine().refined(itr).element_finder()
            self.assertEqual(finder(np.array([0.001]))[0], 0)
            self.assertEqual(finder(np.array([0.999]))[0], 2 ** itr - 1)


class TestFinder1DLinspaced(TestCase):

    def runTest(self):

        for itr in range(5):
            finder = (
                MeshLine(np.linspace(0, 1, 2 ** itr + 1)).element_finder()
            )
            self.assertEqual(finder(np.array([0.999]))[0], 2 ** itr - 1)
            self.assertEqual(finder(np.array([0.001]))[0], 0)


@pytest.mark.parametrize(
    "m",
    [
        MeshTri.init_circle(),
        MeshQuad.init_tensor([0, 1, 3], [0, 1, 3]),
        MeshTet().refined(3),
        MeshHex.init_tensor([0, 1, 3], [0, 1, 3], [0, 1, 3]),
    ]
)   
def test_smoothed(m):
    M = m.smoothed()
    assert M.is_valid()
    # points have moved?
    assert np.linalg.norm((M.p - m.p) ** 2) > 0


@pytest.mark.parametrize(
    "m",
    [
        MeshTri(),
        MeshTet(),
    ]
)
def test_oriented(m):
    M = m.oriented()
    assert np.sum(m.orientation() < 0) > 0
    assert np.sum(m.orientation() > 0) > 0
    assert np.sum(M.orientation() > 0) > 0
    assert np.sum(M.orientation() < 0) == 0


@pytest.mark.parametrize(
    "m,seed",
    [
        (MeshTri(), 0),
        (MeshTri(), 1),
        (MeshTri().refined(), 2),
        (MeshTet(), 0),
        (MeshTet(), 1),
        (MeshTet(), 2),
        (MeshTet().refined(), 10),
    ]
)
def test_finder_simplex(m, seed):

    np.random.seed(seed)
    points = np.hstack((m.p, np.random.rand(m.p.shape[0], 100)))
    tri = Delaunay(points.T)
    M = type(m)(points, tri.simplices.T)
    finder = M.element_finder()
    
    query_pts = np.random.rand(m.p.shape[0], 500)
    assert_array_equal(
        tri.find_simplex(query_pts.T),
        finder(*query_pts),
    )


@pytest.mark.parametrize(
    "m",
    [
        MeshTri(),
        MeshQuad(),
        MeshTet(),
        MeshHex(),
        MeshTri2(),
        MeshQuad2(),
        MeshTet2(),
        MeshHex2(),
        MeshLine(),
    ]
)
def test_meshio_cycle(m):

    M = from_meshio(to_meshio(m))
    assert_array_equal(M.p, m.p)
    assert_array_equal(M.t, m.t)
    if m.boundaries is not None:
        np.testing.assert_equal(m.boundaries, M.boundaries)
    if m.subdomains is not None:
        np.testing.assert_equal(m.subdomains, M.subdomains)


_test_lambda = {
    'left': lambda x: x[0] < 0.6,
    'right': lambda x: x[0] > 0.3,
}


@pytest.mark.parametrize(
    "boundaries_only",
    [
        True,
        False,
    ]
)
@pytest.mark.parametrize(
    "m",
    [
        MeshTri(),
        MeshQuad(),
        MeshHex(),
        MeshTet(),
        MeshHex().refined(),
        MeshTet.load(MESH_PATH / 'box.msh'),
        MeshTri.load(MESH_PATH / 'square.msh'),
        MeshTet2.load(MESH_PATH / 'quadraticsphere.msh'),
    ]
)
def test_meshio_cycle_boundaries(boundaries_only, m):

    m = m.with_boundaries(_test_lambda, boundaries_only)
    M = from_meshio(to_meshio(m))
    assert_array_equal(M.p, m.p)
    assert_array_equal(M.t, m.t)
    for key in m.boundaries:
        assert_array_equal(M.boundaries[key].sort(),
                           m.boundaries[key].sort())


@pytest.mark.parametrize(
    "m",
    [
        MeshTri(),
        MeshQuad(),
        MeshHex(),
        MeshTet(),
    ]
)
def test_meshio_cycle_subdomains(m):

    m = m.refined(2).with_subdomains(_test_lambda)
    M = from_meshio(to_meshio(m))
    assert_array_equal(M.p, m.p)
    assert_array_equal(M.t, m.t)
    for key in m.subdomains:
        assert_array_equal(M.subdomains[key], m.subdomains[key])


@pytest.mark.parametrize(
    "m",
    [
        MeshTri(),
        MeshQuad(),
        MeshTet(),
        MeshHex(),
    ]
)
def test_saveload_cycle_vtk(m):

    from tempfile import NamedTemporaryFile
    m = m.refined(2)
    with NamedTemporaryFile(suffix='.vtk') as f:
        m.save(f.name)
        m2 = Mesh.load(f.name)

    assert_array_equal(m.p, m2.p)
    assert_array_equal(m.t, m2.t)


@pytest.mark.parametrize(
    "fmt, kwargs",
    [
        ('.msh', {}),
        ('.msh', {'file_format': 'gmsh22'}),
        ('.vtk', {}),
        #('.xdmf', {}),
        ('.vtu', {}),
        #('.med', {}),
    ]
)
@pytest.mark.parametrize(
    "m",
    [
        MeshTri(),
        MeshQuad(),
        MeshHex(),  # TODO facet order changes?
        MeshTet(),
    ]
)
def test_saveload_cycle_tags(fmt, kwargs, m):

    m = (m
         .refined(2)
         .with_subdomains(_test_lambda)
         .with_boundaries({'test': lambda x: (x[0] == 0) * (x[1] < 0.6),
                           'set': lambda x: (x[0] == 0) * (x[1] > 0.3)}))
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=fmt) as f:
        m.save(f.name, point_data={'foo': m.p[0]}, **kwargs)
        out = ['point_data', 'cells_dict']
        m2 = Mesh.load(f.name, out=out)


        assert_array_equal(m.p, m2.p)
        assert_array_equal(m.t, m2.t)
        assert_array_equal(out[0]['foo'], m.p[0])
        for key in m.subdomains:
            assert_array_equal(m2.subdomains[key].sort(),
                               m.subdomains[key].sort())
        for key in m.boundaries:
            assert_array_equal(m2.boundaries[key].sort(),
                               m.boundaries[key].sort())


def test_periodic_failure():

    # these meshes cannot be made periodic due to insufficient number of
    # elements
    with pytest.raises(ValueError):
        mp = MeshLine1DG.periodic(MeshLine(), [0], [1])

    with pytest.raises(ValueError):
        mp = MeshQuad1DG.periodic(MeshQuad(), [0], [1])

    with pytest.raises(ValueError):
        mp = MeshQuad1DG.periodic(MeshQuad().refined(2), [0], [1, 2])


@pytest.mark.parametrize(
    "mtype",
    [
        MeshTri,
        MeshQuad,
        MeshHex,
        MeshTet,
        MeshLine,
    ]
)
def test_init_refdom(mtype):

    m = mtype.init_refdom()
    mapping = m._mapping()
    x = mapping.F(m.p)[:, 0, :]
    assert_array_equal(x, m.p)


@pytest.mark.parametrize(
    "mtype",
    [
        MeshTri,
        MeshQuad,
        MeshLine,
    ]
)
def test_refine_boundaries(mtype):

    morig = mtype().refined()

    m = morig.with_boundaries({'test1': lambda x: x[0] == 0,
                               'test2': lambda x: x[0] == 1})
    M1 = m.refined()
    M2 = morig.refined().with_boundaries({'test1': lambda x: x[0] == 0,
                                          'test2': lambda x: x[0] == 1})

    # check that same facets exist no matter the order of with_boundaries
    # and refined
    np.testing.assert_equal(M1.boundaries, M2.boundaries)


def test_point_outside_mesh():

    m = MeshTri.load(MESH_PATH / 'troublesome_mesh.vtk')
    elem_finder = m.element_finder()
    elem_finder(*m.p)


def test_refine_subdomains_adaptive():

    sdef = {'left': lambda x: x[0] < 0.5}

    for inds in [
            [1,2,3,4,5,10,25],
            [10,20,30,40,50],
            [10,20,30,31,32,33,34,40,50],
    ]:
        m1 = MeshTri().refined(3).with_subdomains(sdef).refined(inds)
        m2 = MeshTri().refined(3).refined(inds).with_subdomains(sdef)
        np.testing.assert_equal(m1.subdomains, m2.subdomains)


def test_refine_subdomains_uniform():

    sdef = {'left': lambda x: x[0] < 0.5,
            'top': lambda x: x[1] > 0.5}

    m1 = MeshTri().refined(3).with_subdomains(sdef).refined()
    m2 = MeshTri().refined(3).refined().with_subdomains(sdef)
    np.testing.assert_equal(m1.subdomains, m2.subdomains)


def test_refine_subdomains_uniform_tets():

    sdef = {'left': lambda x: x[0] < 0.5,
            'top': lambda x: x[1] > 0.5}

    m1 = MeshTet().refined().with_subdomains(sdef).refined()
    m2 = MeshTet().refined().refined().with_subdomains(sdef)
    np.testing.assert_equal(m1.subdomains, m2.subdomains)


def test_refine_subdomains_uniform_hexs():

    sdef = {'left': lambda x: x[0] < 0.5,
            'top': lambda x: x[1] > 0.5}

    m1 = MeshHex().refined().with_subdomains(sdef).refined()
    m2 = MeshHex().refined().refined().with_subdomains(sdef)
    np.testing.assert_equal(m1.subdomains, m2.subdomains)
