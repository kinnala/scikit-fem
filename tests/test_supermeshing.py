import numpy as np
import pytest

from skfem.mesh import MeshTri, MeshQuad, MeshTet
from skfem.supermeshing import intersect


def _volumes(m):
    dim = m.p.shape[0]
    return (np.abs(m._mapping().detDF(np.zeros((dim, 1)))[:, 0])
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
