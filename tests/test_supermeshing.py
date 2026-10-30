import numpy as np
import pytest

from skfem import BilinearForm, FacetBasis
from skfem.mesh import (MeshLine, MeshTri, MeshTri2, MeshQuad, MeshTet,
                        MeshTet2, MeshHex, MeshHex2)
from skfem.supermeshing import (intersect, mortar_quadrature,
                                elementwise_quadrature)


def _volumes(m):
    dim = m.p.shape[0]
    return (np.abs(m.mapping().detDF(np.zeros((dim, 1)))[:, 0])
            / (2 if dim == 2 else 6))


def _perturbed(m, seed=0, scale=.01):
    rng = np.random.default_rng(seed)
    p = m.p.copy()
    ix = m.interior_nodes()
    p[:, ix] += scale * rng.standard_normal((p.shape[0], len(ix)))
    return type(m)(p, m.t)


@pytest.mark.parametrize(
    "m1,m2",
    [
        (MeshTri().refined(3), MeshTri().refined(3)),
        (MeshTri().refined(4), MeshTri().refined(2)),
        (_perturbed(MeshTri().refined(4)),
         MeshTri.init_symmetric().refined(3)),
        (MeshTri().refined(3), MeshQuad().refined(3)),
        (MeshQuad().refined(2), _perturbed(MeshTri().refined(3), seed=1)),
        (MeshTet().refined(2), MeshTet().refined(2)),
        (MeshTet().refined(3), MeshTet().refined(1)),
        (_perturbed(MeshTet().refined(2)),
         MeshTet.init_tensor(*(3 * (np.linspace(0, 1, 5),)))),
    ]
)
def test_intersect_covers(m1, m2):
    """Supermesh of two meshes of the unit square/cube covers each element
    of both meshes exactly."""
    m12, ix1, ix2 = intersect(m1, m2)
    vol = _volumes(m12)
    assert np.all(vol > 0)
    assert np.isclose(vol.sum(), 1.)
    if m1.t.shape[0] == m1.p.shape[0] + 1:
        assert np.allclose(np.bincount(ix1, vol, m1.t.shape[1]),
                           _volumes(m1))
    if m2.t.shape[0] == m2.p.shape[0] + 1:
        assert np.allclose(np.bincount(ix2, vol, m2.t.shape[1]),
                           _volumes(m2))
    # supermesh elements lie inside their parent elements
    for m, ix in ((m1, ix1), (m2, ix2)):
        X = m.mapping().invF(m12.p[:, m12.t].transpose(0, 2, 1), tind=ix)
        assert np.all(X > -1e-10) and np.all(X < 1 + 1e-10)


@pytest.mark.parametrize(
    "m1,m2,overlap",
    [
        (MeshTri().refined(3),
         MeshTri().refined(2).translated((.5, .25)), .5 * .75),
        (MeshTet().refined(2),
         MeshTet().refined(3).translated((.5, .5, .5)), .125),
        (MeshTet().refined(2),
         MeshTet().translated((2., 0., 0.)), 0.),
    ]
)
def test_intersect_partial_overlap(m1, m2, overlap):
    m12, _, _ = intersect(m1, m2)
    assert np.isclose(_volumes(m12).sum(), overlap)


def _mortar_bases(m1, f1, m2, f2, max_dist=.5, **kwargs):
    q1, f1, q2, f2 = mortar_quadrature(m1, f1, m2, f2, max_dist, **kwargs)
    fb1 = FacetBasis(m1, m1.elem(), quadrature=q1, facets=f1)
    fb2 = FacetBasis(m2, m2.elem(), quadrature=q2, facets=f2)
    return (fb1, fb2,
            np.array(fb1.global_coordinates()),
            np.array(fb2.global_coordinates()))


def _boundary(m, f):
    return np.intersect1d(m.boundary_facets(), m.facets_satisfying(f))


_MASS = [
    BilinearForm(lambda u1, u2, v1, v2, w: u1 * v1),
    BilinearForm(lambda u1, u2, v1, v2, w: u2 * v1),
    BilinearForm(lambda u1, u2, v1, v2, w: u2 * v2),
]


@pytest.mark.parametrize("slave", [1, 2])
@pytest.mark.parametrize(
    "m1,f1,m2,f2,gap,tmesh1,tmesh2",
    [
        # 2D with a gap
        (MeshTri.init_tensor(np.linspace(0, 1, 8), np.linspace(.1, .5, 3)),
         lambda x: x[1] < .1 + 1e-9,
         MeshQuad.init_tensor(np.linspace(-.2, 1.2, 11),
                              np.linspace(-.5, 0, 3)),
         lambda x: x[1] > -1e-9,
         .1, MeshLine, MeshLine),
        # 2D with penetration, partial overlap
        (MeshQuad.init_tensor(np.linspace(0, 1, 8), np.linspace(-.1, .5, 3)),
         lambda x: x[1] < -.1 + 1e-9,
         MeshTri.init_tensor(np.linspace(.3, 1.2, 6),
                             np.linspace(-.5, 0, 3)),
         lambda x: x[1] > -1e-9,
         -.1, MeshLine, MeshLine),
        # 3D tetrahedra on hexahedra
        (MeshTet.init_tensor(*(2 * (np.linspace(0, 1, 5),)),
                             np.linspace(.05, .5, 3)),
         lambda x: x[2] < .05 + 1e-9,
         MeshHex.init_tensor(*(2 * (np.linspace(-.1, 1.13, 7),)),
                             np.linspace(-.5, 0, 3)),
         lambda x: x[2] > -1e-9,
         .05, MeshTri, MeshQuad),
        # 3D hexahedra on tetrahedra, touching, partial overlap
        (MeshHex.init_tensor(*(2 * (np.linspace(0, 1, 4),)),
                             np.linspace(0, .5, 3)),
         lambda x: x[2] < 1e-9,
         MeshTet.init_tensor(*(2 * (np.linspace(.2, 1.13, 6),)),
                             np.linspace(-.5, 0, 3)),
         lambda x: x[2] > -1e-9,
         0., MeshQuad, MeshTri),
    ]
)
def test_mortar_quadrature_flat(m1, f1, m2, f2, gap, tmesh1, tmesh2, slave):
    """Compare against the supermesh of the projected trace meshes."""
    f1, f2 = _boundary(m1, f1), _boundary(m2, f2)
    fb1, fb2, x1, x2 = _mortar_bases(m1, f1, m2, f2, slave=slave)
    assert np.allclose(fb1.dx, fb2.dx)
    # the paired points are separated by the gap along the normal
    assert np.allclose(x2 - x1, gap * np.array(fb1.normals))

    def project(p):
        return p[:-1]

    t1, o1 = m1.trace(f1, mtype=tmesh1, project=project)
    t2, o2 = m2.trace(f2, mtype=tmesh2, project=project)
    m12, ix1, ix2 = intersect(t1, t2)
    ref = (FacetBasis(m1, m1.elem(),
                      quadrature=elementwise_quadrature(t1, m12, ix1),
                      facets=o1[ix1])
           * FacetBasis(m2, m2.elem(),
                        quadrature=elementwise_quadrature(t2, m12, ix2),
                        facets=o2[ix2]))
    for form in _MASS:
        assert np.allclose(form.assemble(fb1 * fb2).toarray(),
                           form.assemble(ref).toarray(),
                           atol=1e-14)


def _annulus(r1, r2, n):
    m = MeshQuad.init_tensor(np.linspace(r1, r2, 3),
                             np.linspace(0, 2 * np.pi, n + 1))
    return MeshQuad(np.array([m.p[0] * np.cos(m.p[1]),
                              m.p[0] * np.sin(m.p[1])]), m.t)


def test_mortar_quadrature_annulus():
    m1 = MeshTri.init_circle(4)
    m2 = _annulus(1.2, 1.5, 100)
    inner = m2.facets_satisfying(
        lambda x: np.abs(np.sqrt(x[0] ** 2 + x[1] ** 2) - 1.2) < 1e-2
    )
    fb1, fb2, x1, x2 = _mortar_bases(m1, m1.boundary_facets(), m2, inner)
    assert np.allclose(fb1.dx, fb2.dx)
    # slave boundary is covered exactly
    assert np.isclose(fb1.dx.sum(), FacetBasis(m1, m1.elem()).dx.sum())
    # master points lie on the inner circle (polygon)
    assert np.allclose(np.linalg.norm(x2, axis=0), 1.2, atol=1e-3)


@pytest.mark.parametrize(
    "m1,f1,m2,f2",
    [
        (MeshTri2.init_circle(3),
         lambda x: x[1] < -.5,
         MeshTri2.from_mesh(MeshTri.init_tensor(np.linspace(-2, 2, 9),
                                                np.linspace(-2, -1, 3))),
         lambda x: x[1] > -1 - 1e-9),
        (MeshTet.init_ball(3),
         lambda x: x[2] < -.5,
         MeshTet.init_tensor(*(2 * (np.linspace(-3, 3, 9),)),
                             np.linspace(-2, -1, 3)),
         lambda x: x[2] > -1 - 1e-9),
    ]
)
def test_mortar_quadrature_curved_slave(m1, f1, m2, f2):
    f1 = _boundary(m1, f1)
    fb1, fb2, x1, x2 = _mortar_bases(m1, f1, m2, _boundary(m2, f2),
                                     max_dist=2.)
    assert np.isclose(fb1.dx.sum(),
                      FacetBasis(m1, m1.elem(), facets=f1,
                                 intorder=6).dx.sum())
    assert np.allclose(x2[-1], -1.)


def test_mortar_quadrature_curved_hex():
    """Master points on bilinear facets of a distorted hexahedral mesh."""
    m1 = MeshTet.init_tensor(*(2 * (np.linspace(0, 1, 5),)),
                             np.linspace(.2, .5, 3))
    m2 = MeshHex.init_tensor(*(2 * (np.linspace(-.1, 1.1, 6),)),
                             np.linspace(-.5, 0, 3))
    p = m2.p.copy()
    top = p[2] == 0
    p[2, top] = .05 * np.sin(3 * p[0, top]) * np.cos(2 * p[1, top])
    m2 = MeshHex(p, m2.t)
    fb1, fb2, x1, x2 = _mortar_bases(
        m1, _boundary(m1, lambda x: x[2] < .2 + 1e-9),
        m2, _boundary(m2, lambda x: x[2] > -.1),
    )
    assert np.isclose(fb1.dx.sum(), 1.)
    assert np.allclose(x2[:2], x1[:2])
    X = m2.mapping().invF(x2, tind=m2.f2t[0, fb2.find])
    assert np.all((X > -1e-8) & (X < 1 + 1e-8))
    assert np.allclose(np.min(np.minimum(X, 1 - X), axis=0), 0.)


def test_mortar_quadrature_max_dist():
    m1 = MeshTri.init_tensor(np.linspace(0, 1, 5), np.linspace(.1, .5, 3))
    m2 = MeshTri.init_tensor(np.linspace(0, 1, 3), np.linspace(-.5, 0, 3))
    f1 = _boundary(m1, lambda x: x[1] < .1 + 1e-9)
    f2 = _boundary(m2, lambda x: x[1] > -1e-9)
    q1, ff1, q2, ff2 = mortar_quadrature(m1, f1, m2, f2, .05)
    assert len(ff1) == 0 and len(ff2) == 0
    # the bottom of the master is not facing the slave
    q1, ff1, q2, ff2 = mortar_quadrature(
        m1, f1, m2, _boundary(m2, lambda x: x[1] < -.5 + 1e-9), 1.
    )
    assert len(ff1) == 0
    # 3D, too far
    m1 = MeshTet().translated((0., 0., 2.))
    m2 = MeshHex()
    q1, ff1, q2, ff2 = mortar_quadrature(m1, m1.boundary_facets(),
                                         m2, m2.boundary_facets(), .1)
    assert len(ff1) == 0 and q1[0].shape[1] == 0


def test_mortar_quadrature_swap():
    """Swapping the slave side changes the segmentation of curved
    boundaries but keeps the outputs in the order of the arguments."""
    m1 = MeshTri.init_circle(4)
    m2 = _annulus(1.2, 1.5, 100)
    f1 = m1.boundary_facets()
    f2 = m2.facets_satisfying(
        lambda x: np.abs(np.sqrt(x[0] ** 2 + x[1] ** 2) - 1.2) < 1e-2
    )
    fb1, fb2, x1, x2 = _mortar_bases(m1, f1, m2, f2, slave=2)
    assert np.all(np.isin(fb1.find, f1)) and np.all(np.isin(fb2.find, f2))
    assert np.allclose(fb1.dx, fb2.dx)
    # the master side (m2) is now covered exactly
    assert np.isclose(fb2.dx.sum(),
                      FacetBasis(m2, m2.elem(), facets=f2).dx.sum())
    assert np.allclose(np.linalg.norm(x1, axis=0), 1., atol=2e-3)
    q1, _, _, _ = mortar_quadrature(m1, f1, m2, f2, .5)
    assert not np.isclose(q1[1].sum(), fb1.W.sum())
    with pytest.raises(ValueError):
        mortar_quadrature(m1, f1, m2, f2, .5, slave=3)


def _inside(m, X, tol=1e-8):
    if m.brefdom.nnodes == m.dim():
        return np.all(X > -tol) and np.all(X.sum(axis=0) < 1 + tol)
    return np.all((X > -tol) & (X < 1 + tol))


def test_mortar_quadrature_quadratic_ball_on_plane():
    """Curved slave facets are covered exactly."""
    m1 = MeshTet2.init_ball(2)
    m2 = MeshTet2.from_mesh(MeshTet.init_tensor(
        *(2 * (np.linspace(-1.5, 1.5, 7),)), np.linspace(-2, -1.05, 3)))
    f1 = _boundary(m1, lambda x: x[2] < -.3)
    fb1, fb2, x1, x2 = _mortar_bases(
        m1, f1, m2, _boundary(m2, lambda x: x[2] > -1.05 - 1e-9), max_dist=2.
    )
    assert np.allclose(fb1.dx, fb2.dx)
    assert np.allclose(x2[2], -1.05)
    # facets whose normal rays surely hit the plane are covered exactly
    fc = _boundary(m1, lambda x: x[2] < -.8)
    assert np.isclose(fb1.dx[np.isin(fb1.find, fc)].sum(),
                      FacetBasis(m1, m1.elem(), facets=fc,
                                 intorder=10).dx.sum(),
                      rtol=1e-12)


def test_mortar_quadrature_quadratic_plane_on_ball():
    """Master points on curved facets are moved to the facet they hit."""
    m1 = MeshTet2.from_mesh(MeshTet.init_tensor(
        *(2 * (np.linspace(-.5, .5, 5),)), np.linspace(1.05, 1.5, 3)))
    m2 = MeshTet2.init_ball(2)
    f1 = _boundary(m1, lambda x: x[2] < 1.05 + 1e-9)
    fb1, fb2, x1, x2 = _mortar_bases(
        m1, f1, m2, _boundary(m2, lambda x: x[2] > .3), max_dist=1.
    )
    assert np.allclose(fb1.dx, fb2.dx)
    # the slave plane is covered exactly
    assert np.isclose(fb1.dx.sum(), 1.)
    # straight down from the slave points
    assert np.allclose(x2[:2], x1[:2])
    # on the P2 sphere up to the geometric error
    assert np.allclose(np.linalg.norm(x2, axis=0), 1., atol=5e-3)
    assert _inside(m2, np.array(fb2.X))


def test_mortar_quadrature_quadratic_ball_on_ball():
    m1 = MeshTet2.init_ball(2)
    m2 = MeshTet2.init_ball(2).translated((2.05, 0., 0.))
    fb1, fb2, x1, x2 = _mortar_bases(
        m1, _boundary(m1, lambda x: x[0] > .5),
        m2, _boundary(m2, lambda x: x[0] < 1.55),
    )
    assert np.allclose(fb1.dx, fb2.dx)
    assert _inside(m1, np.array(fb1.X)) and _inside(m2, np.array(fb2.X))
    # points are paired along the normals of the curved slave facets
    t = (x2 - x1).reshape(3, -1)
    n = np.array(fb1.normals).reshape(3, -1)
    cos = np.einsum('ij,ij->j', t, n) / np.linalg.norm(t, axis=0)
    assert np.allclose(cos, 1.)


def test_mortar_quadrature_quadratic_hex():
    m1 = MeshHex2.from_mesh(MeshHex.init_tensor(
        *(2 * (np.linspace(0, 1, 4),)), np.linspace(.1, .5, 3)))
    m2 = MeshHex2.from_mesh(MeshHex.init_tensor(
        *(2 * (np.linspace(-.1, 1.1, 5),)), np.linspace(-.5, 0, 3)))
    p = m2.doflocs.copy()
    top = p[2] > -1e-9
    p[2, top] = .05 * np.sin(3 * p[0, top]) * np.cos(2 * p[1, top])
    m2 = MeshHex2(p, m2.t)
    fb1, fb2, x1, x2 = _mortar_bases(
        m1, _boundary(m1, lambda x: x[2] < .1 + 1e-9),
        m2, _boundary(m2, lambda x: x[2] > -.1),
    )
    assert np.allclose(fb1.dx, fb2.dx)
    assert np.isclose(fb1.dx.sum(), 1.)
    assert np.allclose(x2[:2], x1[:2])
    assert _inside(m2, np.array(fb2.X))


@pytest.mark.parametrize("nsub", [1, 2, 3])
def test_mortar_quadrature_nsub(nsub):
    """Flat facets give the same mortar matrices for any subdivision."""
    m1 = MeshTet.init_tensor(*(2 * (np.linspace(0, 1, 4),)),
                             np.linspace(.05, .5, 3))
    m2 = MeshHex.init_tensor(*(2 * (np.linspace(-.1, 1.13, 5),)),
                             np.linspace(-.5, 0, 3))
    f1 = _boundary(m1, lambda x: x[2] < .05 + 1e-9)
    f2 = _boundary(m2, lambda x: x[2] > -1e-9)
    fb = _mortar_bases(m1, f1, m2, f2)
    fbn = _mortar_bases(m1, f1, m2, f2, nsub=nsub)
    for form in _MASS:
        assert np.allclose(form.assemble(fb[0] * fb[1]).toarray(),
                           form.assemble(fbn[0] * fbn[1]).toarray(),
                           atol=1e-14)
