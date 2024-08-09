from unittest import TestCase
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import Delaunay
from numpy.testing import assert_array_equal, assert_almost_equal

from skfem.mesh import (Mesh, MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri,
                        MeshTet1, MeshHex1, MeshLine1DG, MeshQuad1DG, 
                        MeshTri1DG, MeshTri2, MeshQuad2, MeshTet2, MeshHex2)
from skfem.assembly import Basis, LinearForm, Functional, FacetBasis
from skfem.element import (ElementTetP1, ElementTriP0, ElementQuad0,
                           ElementHex0, ElementTriP1)
from skfem.utils import projection
from skfem.io.meshio import to_meshio, from_meshio
from skfem.helpers import dot


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
            M = cls.from_dict(m.to_dict())
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
        m = m.refined(np.arange(m.nelements, dtype=np.int32))
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
    "mtype, path, ignore_orientation",
    [
        (MeshTet, MESH_PATH / 'box.msh', False),
        (MeshTet, MESH_PATH / 'box.msh', True),
        (MeshTri, MESH_PATH / 'tagged_gmsh4.msh', False),
        (MeshTri, MESH_PATH / 'tagged_gmsh4.msh', True),
    ]
)
def test_load_file(mtype, path, ignore_orientation):

    m = mtype.load(path, ignore_orientation=ignore_orientation)
    assert len(m.boundaries) > 0


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
        ('.msh', {'file_format': 'gmsh'}),
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
@pytest.mark.parametrize(
    "ignore_orientation",
    [
        True,
        False,
    ]
)
@pytest.mark.parametrize(
    "ignore_interior_facets",
    [
        True,
        False,
    ]
)
def test_saveload_cycle_tags(fmt, kwargs, m, ignore_orientation, ignore_interior_facets):

    m = (m
         .refined(2)
         .with_subdomains(_test_lambda)
         .with_boundaries({'test': lambda x: (x[0] == 0) * (x[1] < 0.6),
                           'set': lambda x: (x[0] == 0) * (x[1] > 0.3)}))
    from tempfile import NamedTemporaryFile
    with NamedTemporaryFile(suffix=fmt) as f:
        m.save(f.name, point_data={'foo': m.p[0]}, **kwargs)
        out = ['point_data', 'cells_dict']
        m2 = Mesh.load(f.name,
                       out=out,
                       ignore_orientation=ignore_orientation,
                       ignore_interior_facets=ignore_interior_facets)


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


@pytest.mark.parametrize(
    "fbasis,refval,dec",
    [
        (FacetBasis(MeshTri2.init_circle(), ElementTriP0()), 2 * np.pi, 5),
        (FacetBasis(MeshQuad2(), ElementQuad0()), 4, 5),
        (FacetBasis(MeshTet2.init_ball(), ElementTetP1()), 4 * np.pi, 3),
        (FacetBasis(MeshHex2(), ElementHex0()), 6, 3),
    ]
)
def test_integrate_quadratic_boundary(fbasis, refval, dec):

    @Functional
    def unit(w):
        return 1 + 0 * w.x[0]

    np.testing.assert_almost_equal(unit.assemble(fbasis),
                                   refval, decimal=dec)


def test_remove_unused():

    m = MeshTri()
    m1 = MeshTri(m.p, m.t[:, [0]])

    assert not m1.is_valid()
    assert m1.remove_unused_nodes().is_valid()


def test_remove_duplicates():

    m = MeshTri()
    t = m.t
    t[1, :] += 4
    m1 = MeshTri(np.hstack((m.p, m.p)), t)

    assert not m1.is_valid()
    assert m1.remove_duplicate_nodes().is_valid()


@pytest.mark.parametrize(
    "mesh",
    [
        MeshTri().refined(3),
        MeshTet().refined(3),
        MeshQuad().refined(3),
        MeshHex().refined(3),
    ]
)
def test_incidence(mesh):

    p2t = mesh.p2t
    for itr in range(0, 50, 3):
        assert np.sum((mesh.t == itr).any(axis=0)) == len(p2t[:, itr].data)

    p2f = mesh.p2f
    for itr in range(0, 50, 3):
        assert np.sum((mesh.facets == itr).any(axis=0)) == len(p2f[:, itr].data)

    if isinstance(mesh, (MeshTet1, MeshHex1)):
        p2e = mesh.p2e
        for itr in range(0, 50, 3):
            assert np.sum((mesh.edges == itr).any(axis=0)) == len(p2e[:, itr].data)

        e2t = mesh.e2t
        for itr in range(0, 50, 3):
            e = mesh.edges[:, itr]
            datalen = np.sum(
                (mesh.t == e[0]).any(axis=0) & (mesh.t == e[1]).any(axis=0)
            )
            assert datalen == len(e2t[:, itr].data)

def test_restrict_tags_boundary():

    m = MeshTri().refined(3).with_defaults()
    m = m.with_subdomains({
        'left': lambda x: x[0] <= 0.5,
        'bottom': lambda x: x[1] <= 0.5,
    })

    mr = m.restrict('left')

    # test boundary retag
    topleftp = m.p[0, np.unique(m.facets[:, m.boundaries['top']].flatten())]
    topleftp = np.sort(topleftp[topleftp <= 0.5])
    topleftpr = np.sort(mr.p[0, np.unique(mr.facets[:, mr.boundaries['top']].flatten())])

    assert_array_equal(topleftp, topleftpr)


def test_restrict_tags_subdomain():

    m = MeshTri().refined(3)
    m = m.with_subdomains({
        'left': lambda x: x[0] <= 0.5,
        'bottom': lambda x: x[1] <= 0.5,
    })

    mr = m.restrict('left')

    # test subdomain retag
    bottomleftp = m.p[:, np.unique(m.t[:, m.subdomains['bottom']].flatten())]
    bottomleftp = bottomleftp[:, bottomleftp[0] <= 0.5]
    ix = np.argsort(bottomleftp[0] + 0.1 * bottomleftp[1])
    bottomleftp = bottomleftp[:, ix]

    bottomleftpr = mr.p[:, np.unique(mr.t[:, mr.subdomains['bottom']].flatten())]
    ix = np.argsort(bottomleftpr[0] + 0.1 * bottomleftpr[1])
    bottomleftpr = bottomleftpr[:, ix]
    
    assert_array_equal(bottomleftp, bottomleftpr)


def test_restrict_reverse_map():

    m = MeshTri().refined(3)
    m = m.with_subdomains({
        'left': lambda x: x[0] <= 0.5,
        'bottom': lambda x: x[1] <= 0.5,
    })

    mr, ix = m.restrict('left', return_mapping=True)


    p1 = mr.p
    I = np.argsort(p1[0] + 0.1 * p1[1])
    p1 = p1[:, I]

    p2 = m.p[:, ix]
    I = np.argsort(p2[0] + 0.1 * p2[1])
    p2 = p2[:, I]

    assert_array_equal(p1, p2)


HELPER_TETMESH = MeshTet().refined(2).with_subdomains({'sub': lambda x: ((x[0] < 0.75)
                                                                         * (x[1] < 0.75)
                                                                         * (x[2] < 0.75)
                                                                         * (x[0] > 0.25)
                                                                         * (x[1] > 0.25)
                                                                         * (x[2] > 0.25))})


@pytest.mark.parametrize(
    "mesh, volume, etype",
    [
        (MeshTri.load(MESH_PATH / 'oriented_squares.msh'), 0.2 ** 2, ElementTriP1),
        (HELPER_TETMESH.with_boundaries({'subif': HELPER_TETMESH.facets_around('sub')}),
         0.5 ** 3, ElementTetP1),
        (MeshTet.load(MESH_PATH / 'cube_oriented_sub.msh'),
         0.5 ** 3, ElementTetP1),
    ]
)
def test_load_orientation(mesh, volume, etype):

    for k, v in mesh.boundaries.items():
        fbasis = FacetBasis(mesh, etype(), facets=k)

        @Functional
        def form(w):
            # calculate volume using Gauss divergence theorem
            return dot(w.x / 2, w.n)

        np.testing.assert_almost_equal(form.assemble(fbasis),
                                       volume,
                                       1e-10)


def test_hole_orientation():

    # check that a mesh with a tagged hole is oriented correctly

    m = MeshTri.load(MESH_PATH / 'annulus.msh')
    mig = MeshTri.load(MESH_PATH / 'annulus.msh',
                       ignore_orientation=True)

    fbasisint = FacetBasis(m, ElementTriP1(),
                           facets='inter')
    fbasisext = FacetBasis(m, ElementTriP1(),
                           facets='exter')

    fbasisintig = FacetBasis(mig, ElementTriP1(),
                             facets='inter')
    fbasisextig = FacetBasis(mig, ElementTriP1(),
                             facets='exter')

    assert (str(type(m.boundaries['inter']))
            == "<class 'skfem.generic_utils.OrientedBoundary'>")

    np.testing.assert_almost_equal(
        fbasisint.normals,
        fbasisintig.normals,
    )

    np.testing.assert_almost_equal(
        fbasisext.normals,
        fbasisextig.normals,
    )
