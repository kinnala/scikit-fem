import numpy as np
import pytest
from numpy.testing import assert_allclose, assert_array_equal

from skfem import (MeshTet, MeshTet2, MeshHex, MeshHex2, MeshWedge1, MeshTri,
                   Mesh, Basis, EdgeBasis, Functional, LinearForm,
                   ElementTetP1, ElementTetP2, ElementTetN1, ElementHex1,
                   ElementHex2, ElementWedge1, ElementVector)
from skfem.generic_utils import OrientedBoundary
from skfem.helpers import dot
from skfem.mapping import Mapping, MappingAffine
from skfem.mesh._tags import TagKeyError


def axis(x):
    return np.isclose(x[0], 0) & np.isclose(x[1], 0)


def diagonal(x):
    return np.isclose(x[0], x[1]) & np.isclose(x[2], 0)


def bottom(x):
    return np.isclose(x[1], 0) & np.isclose(x[2], 0)


def square(x):
    # the boundary of the unit square at z = 0
    return (np.isclose(x[2], 0)
            & (np.isclose(x[0], 0) | np.isclose(x[0], 1)
               | np.isclose(x[1], 0) | np.isclose(x[1], 1)))


def ccw(x):
    # counterclockwise around the center of the unit square
    return np.array([-(x[1] - .5), x[0] - .5, 0. * x[0]])


MESHES = [
    (MeshTet().refined(2), ElementTetP2()),
    (MeshTet().refined(1), ElementTetP1()),
    (MeshHex().refined(2), ElementHex2()),
    (MeshHex().refined(1), ElementHex1()),
    (MeshTet2().refined(1), ElementTetP2()),
    (MeshHex2().refined(1), ElementHex2()),
    (MeshWedge1(), ElementWedge1()),
]


@pytest.mark.parametrize("m,e", MESHES)
def test_integrate_along_axis(m, e):
    m = m.with_paths({'axis': axis})
    u = Basis(m, e).project(lambda x: x[2] ** 2)
    ebasis = EdgeBasis(m, e, edges='axis')
    expected = 1. / 3. if e.maxdeg >= 2 else None
    val = Functional(lambda w: w['u']).assemble(ebasis, u=u)
    if expected is not None:
        assert_allclose(val, expected)
    # length of the path
    assert_allclose(Functional(lambda w: 1.).assemble(ebasis), 1.)
    # coordinates are on the path
    x = np.asarray(ebasis.global_coordinates())
    assert_allclose(x[0], 0., atol=1e-12)
    assert_allclose(x[1], 0., atol=1e-12)
    # tangents are unit vectors along the axis
    t = np.asarray(ebasis.tangents)
    assert_allclose(np.abs(t[2]), 1.)


def test_prototype():
    # the example in the original prototype
    m = MeshTet().refined(4)
    ebasis = EdgeBasis(m, ElementTetP1(), edges=m.edges_satisfying(axis))
    f = Basis(m, ElementTetP1()).project(lambda x: x[2] ** 3)
    assert_allclose(Functional(lambda w: w['f']).assemble(ebasis, f=f),
                    .25, rtol=1e-3)


def test_length_of_diagonal():
    m = MeshTet().refined(2).with_paths({'diag': diagonal})
    ebasis = EdgeBasis(m, ElementTetP1(), edges='diag')
    assert_allclose(Functional(lambda w: 1.).assemble(ebasis), np.sqrt(2))
    assert_allclose(np.asarray(ebasis.mesh_parameters()).sum(axis=0),
                    np.sqrt(2))


def test_affine_matches_generic_edge_mapping():
    m = MeshTet().refined(2)
    mapping = m.mapping()
    assert isinstance(mapping, MappingAffine)
    X = np.array([[0., .3, 1.]])
    eind = np.array([0, 5, 17, 3])
    assert_allclose(mapping.H(X, eind), Mapping.H(mapping, X, eind))
    assert_allclose(mapping.DH(X, eind), Mapping.DH(mapping, X, eind))
    assert_allclose(mapping.detDH(X, eind), Mapping.detDH(mapping, X, eind))
    # endpoints match mesh.edges
    x = mapping.H(X)
    assert_allclose(x[:, :, 0], m.p[:, m.edges[0]])
    assert_allclose(x[:, :, -1], m.p[:, m.edges[1]])


def test_curved_edges():
    # a quarter of a cylinder with curved quadratic edges
    m = MeshTet().refined(1)
    m2 = MeshTet2.from_mesh(m)
    r, th = 1 + m2.doflocs[0], np.pi / 2 * m2.doflocs[1]
    m2 = MeshTet2(np.array([r * np.cos(th), r * np.sin(th),
                            m2.doflocs[2]]),
                  m2.t)
    m2 = m2.with_paths({
        'arc': lambda x: (np.isclose(np.linalg.norm(x[:2], axis=0), 1.,
                                     atol=.1)
                          & np.isclose(x[2], 0)),
    })
    ebasis = EdgeBasis(m2, ElementTetP2(), edges='arc', intorder=10)
    assert_allclose(Functional(lambda w: 1.).assemble(ebasis),
                    np.pi / 2, rtol=1e-3)
    x = np.asarray(ebasis.global_coordinates())
    assert_allclose(np.linalg.norm(x[:2], axis=0), 1., rtol=1e-2)
    # refinement keeps the path
    assert len(m2.refined().paths['arc']) == 2 * len(m2.paths['arc'])


def test_orientation_and_circulation():
    m = MeshTet().refined(2)
    m = m.with_paths({
        'up': m.edges_satisfying(axis, tangent=[0., 0., 1.]),
        'down': m.edges_satisfying(axis, tangent=[0., 0., -1.]),
        'loop': m.edges_satisfying(square, tangent=ccw),
    })
    assert isinstance(m.paths['up'], OrientedBoundary)

    def tz(edges):
        ebasis = EdgeBasis(m, ElementTetP1(), edges=edges)
        return Functional(lambda w: w.t[2]).assemble(ebasis)

    assert_allclose(tz('up'), 1.)
    assert_allclose(tz('down'), -1.)

    # circulation of u = (-y, x, 0) is 2 * area
    e = ElementVector(ElementTetP1())
    u = Basis(m, e).project(lambda x: np.array([-x[1], x[0], 0. * x[0]]))
    ebasis = EdgeBasis(m, e, edges='loop')
    assert_allclose(Functional(lambda w: dot(w['u'], w.t))
                    .assemble(ebasis, u=u), 2.)

    # tangential component of a Nedelec field
    u = Basis(m, ElementTetN1()).project(
        lambda x: np.array([0. * x[0], 0. * x[0], 1. + 0. * x[0]]))
    ebasis = EdgeBasis(m, ElementTetN1(), edges='up')
    assert_allclose(Functional(lambda w: dot(w['u'], w.t))
                    .assemble(ebasis, u=u), 1.)


def test_linear_form_on_path():
    m = MeshTet().refined(2).with_paths({'axis': axis})
    basis = Basis(m, ElementTetP1())
    ebasis = EdgeBasis(m, ElementTetP1(), edges='axis')
    f = LinearForm(lambda v, w: v).assemble(ebasis)
    assert f.shape == (basis.N,)
    assert_allclose(f.sum(), 1.)
    # nonzero only on the path
    assert_array_equal(np.nonzero(f)[0],
                       np.sort(basis.get_dofs(edges='axis').flatten()))


def test_default_edges_and_errors():
    m = MeshTet().refined(1)
    with pytest.raises(ValueError):
        EdgeBasis(m, ElementTetP1())
    m = m.with_paths({'a': axis, 'b': diagonal})
    ebasis = EdgeBasis(m, ElementTetP1())
    assert_array_equal(ebasis.eind,
                       np.union1d(m.paths['a'], m.paths['b']))
    with pytest.raises(TagKeyError):
        m.paths['c']
    with pytest.raises(ValueError):
        MeshTri().with_paths({'a': lambda x: x[0] == 0})
    with pytest.raises(ValueError):
        EdgeBasis(MeshTri(), ElementTetP1())


def test_with_element_and_project():
    m = MeshTet().refined(2).with_paths({'axis': axis})
    ebasis = EdgeBasis(m, ElementTetP1(), edges='axis')
    ebasis2 = ebasis.with_element(ElementTetP2())
    assert_array_equal(ebasis2.eind, ebasis.eind)
    assert ebasis2.quadrature == ebasis.quadrature
    ebasis2 = EdgeBasis(m, ElementTetP2(), edges='axis')
    x = ebasis2.project(lambda x: x[2] ** 2)
    D = ebasis2.get_dofs(edges='axis').flatten()
    assert_allclose(x[D], ebasis2.doflocs[2, D] ** 2, atol=1e-12)
    assert_allclose(np.delete(x, D), 0.)


@pytest.mark.parametrize(
    "e,nverts,nedges",
    [
        (ElementTetP1(), 5, 0),
        (ElementTetP2(), 5, 4),
        (ElementTetN1(), 0, 4),
    ]
)
def test_get_dofs(e, nverts, nedges):
    m = MeshTet().refined(2).with_paths({'axis': axis})
    basis = Basis(m, e)
    dofs = basis.get_dofs(edges='axis')
    assert len(dofs.flatten()) == nverts + nedges
    assert_array_equal(dofs.flatten(),
                       basis.get_dofs(edges=axis).flatten())
    assert_array_equal(dofs.flatten(),
                       basis.get_dofs(edges=m.paths['axis']).flatten())
    if nverts > 0:
        D = dofs.nodal['u']
        assert_allclose(basis.doflocs[:2, D], 0.)


def test_refined_and_restricted():
    m = MeshTet().refined(1)
    m = m.with_paths({
        'up': m.edges_satisfying(axis, tangent=[0., 0., 1.]),
        'bottom': bottom,
    })
    for mm in [m.refined(), m.refined(2)]:
        assert_array_equal(np.sort(np.asarray(mm.paths['bottom'])),
                           mm.edges_satisfying(bottom))
        assert_array_equal(np.sort(np.asarray(mm.paths['up'])),
                           mm.edges_satisfying(axis))
        ebasis = EdgeBasis(mm, ElementTetP1(), edges='up')
        assert_allclose(Functional(lambda w: w.t[2]).assemble(ebasis), 1.)

    # adaptive
    mm = m.refined(np.arange(0, m.nelements, 2))
    assert_array_equal(np.sort(np.asarray(mm.paths['up'])),
                       mm.edges_satisfying(axis))

    # hexahedral
    mh = MeshHex().refined(1).with_paths({'a': axis})
    assert_array_equal(mh.refined(2).paths['a'],
                       mh.refined(2).edges_satisfying(axis))

    # restrict
    mr = m.refined().restrict(lambda x: x[2] < .5)
    assert_array_equal(np.sort(np.asarray(mr.paths['up'])),
                       mr.edges_satisfying(axis))
    ebasis = EdgeBasis(mr, ElementTetP1(), edges='up')
    assert_allclose(Functional(lambda w: w.t[2]).assemble(ebasis), .5)
    assert len(m.restrict(lambda x: x[2] < .5, skip_paths=True).paths) == 0

    # remove nodes
    assert_array_equal(m.remove_unused_nodes().paths['bottom'],
                       m.paths['bottom'])


def _assert_same_paths(m1, m2, orientation=True):
    assert set(m1.paths) == set(m2.paths)
    for k in m1.paths:
        assert_array_equal(np.asarray(m1.paths[k]), np.asarray(m2.paths[k]))
        if orientation and isinstance(m1.paths[k], OrientedBoundary):
            assert isinstance(m2.paths[k], OrientedBoundary)
            assert_array_equal(m1.paths[k].ori, m2.paths[k].ori)


def _tagged_mesh():
    m = MeshTet().refined(2)
    return m.with_paths({
        'up': m.edges_satisfying(axis, tangent=[0., 0., 1.]),
        'loop': m.edges_satisfying(square, tangent=ccw),
        'diag': diagonal,
    }).with_boundaries({'left': lambda x: x[0] == 0})


def test_save_load_npz(tmp_path):
    m = _tagged_mesh()
    m.save_npz(tmp_path / "mesh.npz")
    m2 = MeshTet.load_npz(tmp_path / "mesh.npz")
    _assert_same_paths(m, m2)
    assert_array_equal(m.boundaries['left'], m2.boundaries['left'])


def test_save_load_npy(tmp_path):
    m = _tagged_mesh()
    m.save_npy(tmp_path / "mesh")
    _assert_same_paths(m, MeshTet.load_npy(tmp_path / "mesh"))


def test_save_load_json(tmp_path):
    from skfem.io.json import to_file, from_file
    m = _tagged_mesh()
    to_file(m, tmp_path / "mesh.json")
    _assert_same_paths(m, from_file(tmp_path / "mesh.json"))


def test_dict_without_paths():
    m = MeshTet()
    d = m.to_dict()
    assert 'paths' not in d
    assert len(MeshTet.from_dict(d).paths) == 0


@pytest.mark.parametrize("m", [_tagged_mesh(),
                               MeshHex().refined(1).with_paths({'a': axis})])
@pytest.mark.parametrize("ext", [".vtk", ".vtu"])
def test_save_load_meshio(tmp_path, m, ext):
    m.save(tmp_path / ("mesh" + ext), encode_point_data=True)
    _assert_same_paths(m, Mesh.load(tmp_path / ("mesh" + ext)),
                       orientation=False)


def test_path_tags():
    from skfem.visuals.web import plot
    m = _tagged_mesh()
    basis, y = m.path_tags()
    assert basis.mesh.p.shape[0] == 3
    edges = np.unique(np.concatenate([np.asarray(v)
                                      for v in m.paths.values()]))
    assert len(y) == basis.N == len(edges)
    mid = basis.mesh.p[:, basis.mesh.t].mean(axis=1)
    for i, k in enumerate(m.paths):
        sel = np.nonzero(y == i + 1)[0]
        emid = m.p[:, m.edges[:, m.paths[k]]].mean(axis=1)
        # the last one overrides the shared edges
        assert len(sel) <= len(m.paths[k])
        for s in sel:
            assert np.isclose(emid, mid[:, s:s + 1]).all(axis=0).any()
    basis, y = m.path_tags(['diag'])
    assert len(y) == len(m.paths['diag'])
    assert (y == 1).all()
    assert plot(*m.path_tags()).payload['dim'] == 1
    with pytest.raises(ValueError):
        MeshTet().path_tags()


def test_repr():
    assert "Named paths [# edges]: up [4]" in repr(_tagged_mesh())
