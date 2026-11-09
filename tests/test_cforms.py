import shutil
import warnings

import numpy as np
import pytest

from skfem import (BilinearForm, LinearForm, Basis, FacetBasis, MeshTri,
                   MeshTet, MeshQuad, MeshHex, MeshLine, ElementTriP2,
                   ElementTetP1, ElementQuad2, ElementTriMini, ElementDG,
                   ElementTetP2, ElementLineP2, ElementVector, ElementTriP1,
                   ElementHex1)
from skfem.helpers import dot, grad, inner, ddot, sym_grad, div
from skfem.models.elasticity import linear_stress

pytestmark = pytest.mark.skipif(shutil.which('cc') is None,
                                reason="No C compiler")

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    from skfem.experimental.cforms import CBilinearForm, CLinearForm


def laplace(u, v, w):
    return dot(grad(u), grad(v)) + 3. * u * v


def nonlinear(u, v, w):
    return ((1. + w.u0 ** 2) * dot(grad(u), grad(v))
            + w.k * np.exp(w.x[0]) * inner(u, v))


def load(v, w):
    return np.sin(np.pi * w.x[0]) * v + w.k * dot(grad(w.u0), grad(v))


def elasticity(u, v, w):
    return (ddot(linear_stress(1., 2.)(sym_grad(u)), sym_grad(v))
            + div(u) * div(v))


SCALAR = [
    (MeshTri().refined(3), ElementTriP2()),
    (MeshTet().refined(2), ElementTetP1()),
    (MeshQuad().refined(3), ElementQuad2()),
    (MeshTri().refined(2), ElementTriMini()),
    (MeshTet().refined(1), ElementDG(ElementTetP2())),
    (MeshLine().refined(3), ElementLineP2()),
]


@pytest.mark.parametrize("m,e", SCALAR)
@pytest.mark.parametrize("nthreads", [0, 1, 3])
def test_scalar(m, e, nthreads):
    basis = Basis(m, e)
    u0 = basis.project(lambda x: x[0] * x[-1])

    A = BilinearForm(laplace).assemble(basis)
    B = CBilinearForm(laplace, nthreads=nthreads).assemble(basis)
    assert abs(A - B).max() < 1e-12

    A = BilinearForm(nonlinear).assemble(basis, u0=u0, k=2.)
    cform = CBilinearForm(nonlinear, nthreads=nthreads)
    B = cform.assemble(basis, u0=u0, k=2.)
    assert abs(A - B).max() < 1e-12

    a = LinearForm(load).assemble(basis, u0=u0, k=2.)
    b = CLinearForm(load, nthreads=nthreads).assemble(basis, u0=u0, k=2.)
    assert np.abs(a - b).max() < 1e-12


@pytest.mark.parametrize("m,e", [
    (MeshTri().refined(3), ElementVector(ElementTriP2())),
    (MeshHex().refined(2), ElementVector(ElementHex1())),
])
def test_vector(m, e):
    basis = Basis(m, e)
    A = BilinearForm(elasticity).assemble(basis)
    B = CBilinearForm(elasticity).assemble(basis)
    assert abs(A - B).max() < 1e-12


def test_decorator_and_elemental():

    @CBilinearForm(nthreads=2)
    def form(u, v, w):
        return dot(grad(u), grad(v))

    assert isinstance(form, CBilinearForm)
    basis = Basis(MeshTri().refined(2), ElementTriP1())
    ref = BilinearForm(form.form)
    assert np.allclose(ref.elemental(basis).tolocal(),
                       form.elemental(basis).tolocal())


def test_mixed_bases():
    m = MeshTri().refined(2)
    ub = Basis(m, ElementVector(ElementTriP2()))
    vb = ub.with_element(ElementTriP1())

    def b(u, v, w):
        return div(u) * v

    A = BilinearForm(b).assemble(ub, vb)
    B = CBilinearForm(b).assemble(ub, vb)
    assert abs(A - B).max() < 1e-12


def test_fallback():
    basis = FacetBasis(MeshTri(), ElementTriP1())
    with pytest.warns(UserWarning, match="falls back"):
        B = CBilinearForm(laplace).assemble(basis)
    assert abs(BilinearForm(laplace).assemble(basis) - B).max() < 1e-12

    def branching(u, v, w):
        return np.where(w.x[0] > .5, 1., 2.) * u * v

    basis = Basis(MeshTri(), ElementTriP1())
    with pytest.warns(UserWarning, match="falls back"):
        B = CBilinearForm(branching).assemble(basis)
    assert abs(BilinearForm(branching).assemble(basis) - B).max() < 1e-12


@pytest.mark.parametrize("m,e", SCALAR)
@pytest.mark.parametrize("nthreads", [1, 2, 7])
def test_cache_sparsity(m, e, nthreads):
    basis = Basis(m, e)
    u0 = basis.project(lambda x: x[0] * x[-1])
    form = CBilinearForm(nonlinear, nthreads=nthreads, cache_sparsity=True)
    A = BilinearForm(nonlinear).assemble(basis, u0=u0, k=2.)
    B = form.assemble(basis, u0=u0, k=2.)
    assert abs(A - B).max() < 1e-12
    # pattern is reused: same index arrays, only the values change
    C = form.assemble(basis, u0=2. * u0, k=3.)
    assert len(form._sparsity) == 1
    assert C.indices is B.indices or np.array_equal(C.indices, B.indices)
    A = BilinearForm(nonlinear).assemble(basis, u0=2. * u0, k=3.)
    assert abs(A - C).max() < 1e-12


def test_cache_sparsity_decorator_vector_subset():

    @CBilinearForm(nthreads=3, cache_sparsity=True)
    def form(u, v, w):
        return elasticity(u, v, w)

    assert form.cache_sparsity
    m = MeshTri().refined(4)
    basis = Basis(m, ElementVector(ElementTriP2()),
                  elements=lambda x: x[0] < .6)
    A = BilinearForm(elasticity).assemble(basis)
    assert abs(A - form.assemble(basis)).max() < 1e-12

    # mixed bases have their own pattern
    vb = Basis(m, ElementTriP1())
    ub = vb.with_element(ElementVector(ElementTriP2()))

    @CBilinearForm(nthreads=3, cache_sparsity=True)
    def b(u, v, w):
        return div(u) * v

    assert abs(BilinearForm(b.form).assemble(ub, vb)
               - b.assemble(ub, vb)).max() < 1e-12


@pytest.mark.parametrize("nthreads", [2, 4, 8])
def test_partition_disjoint_rows(nthreads):
    from skfem.experimental.cforms import _partition
    basis = Basis(MeshTet().refined(3), ElementTetP2())
    perm, parts, phases = _partition(basis, nthreads)
    assert np.array_equal(np.sort(perm), np.arange(basis.nelems))
    for f in range(len(phases) - 1):
        groups = range(phases[f], phases[f + 1])
        rows = [np.unique(basis.element_dofs[:, perm[parts[g]:parts[g + 1]]])
                for g in groups]
        for p in range(len(rows)):
            for q in range(p + 1, len(rows)):
                assert len(np.intersect1d(rows[p], rows[q])) == 0
    # most elements are in the first phase
    assert parts[phases[1]] > .3 * basis.nelems


@pytest.mark.parametrize("nthreads", [1, 4])
@pytest.mark.parametrize("dtype", [None, np.int64])
def test_sparsity_pattern(nthreads, dtype):
    from scipy.sparse import coo_matrix
    from skfem.experimental.cforms import sparsity_pattern
    m = MeshTet().refined(2)
    vb = Basis(m, ElementTetP1(), elements=lambda x: x[1] < .7)
    ub = vb.with_element(ElementVector(ElementTetP2()))
    for u, v in [(vb, vb), (ub, ub), (ub, vb), (vb, ub)]:
        indptr, indices = sparsity_pattern(u.element_dofs, v.element_dofs,
                                           u.N, v.N, nthreads, dtype)
        rows = np.broadcast_to(v.element_dofs[None], (u.Nbfun,)
                               + v.element_dofs.shape)
        cols = np.broadcast_to(u.element_dofs[:, None], (u.Nbfun, v.Nbfun,
                                                         u.nelems))
        K = coo_matrix((np.ones(rows.size), (rows.ravel(), cols.ravel())),
                       shape=(v.N, u.N)).tocsr()
        K.sort_indices()
        assert np.array_equal(indptr, K.indptr)
        assert np.array_equal(indices, K.indices)
        assert indices.dtype == (np.int32 if dtype is None else np.int64)
        assert indptr.dtype == indices.dtype


@pytest.mark.parametrize("m,e", [
    (MeshTri().refined(4), ElementTriP2()),      # affine: strided invDF
    (MeshQuad().refined(4), ElementQuad2()),     # isoparametric
])
@pytest.mark.parametrize("cache_sparsity", [False, True])
def test_element_subset(m, e, cache_sparsity):
    import gc
    basis = Basis(m, e, elements=lambda x: x[0] < .6)
    A = BilinearForm(laplace).assemble(basis)
    form = CBilinearForm(laplace, nthreads=3, cache_sparsity=cache_sparsity)
    for _ in range(3):
        gc.collect()  # temporary arrays must stay alive during the call
        assert abs(A - form.assemble(basis)).max() < 1e-12


@pytest.mark.parametrize("interpolate_in_c", [False, True])
@pytest.mark.parametrize("cache_sparsity", [False, True])
def test_coefficient_interpolation(interpolate_in_c, cache_sparsity):
    m = MeshTri().refined(3)
    ub = Basis(m, ElementVector(ElementTriP2()))
    vb = ub.with_element(ElementTriP1())
    z = ub.project(lambda x: np.array([x[0] * x[1], x[1] ** 2]))
    p = vb.project(lambda x: np.sin(x[0]))

    def bilin(u, v, w):  # vector-valued coefficient, mixed bases
        return (1. + dot(w.z, w.z)) * div(u) * v + ddot(grad(w.z), grad(u)) * v

    def bilin2(u, v, w):  # two scalar coefficients
        return ((w.p ** 2 + w.q) * dot(grad(u), grad(v))
                + dot(grad(w.p), grad(u)) * v)

    def lin(v, w):
        return dot(w.z, v) + ddot(grad(w.z), grad(v))

    def make(cls, f):
        form = cls(f, cache_sparsity=cache_sparsity) if cls is CBilinearForm \
            else cls(f)
        form.interpolate_in_c = interpolate_in_c
        return form

    A = BilinearForm(bilin).assemble(ub, vb, z=z)
    assert abs(A - make(CBilinearForm, bilin).assemble(ub, vb, z=z)).max() \
        < 1e-12
    A = BilinearForm(bilin2).assemble(vb, p=p, q=2. * p)
    B = make(CBilinearForm, bilin2).assemble(vb, p=p, q=2. * p)
    assert abs(A - B).max() < 1e-12
    a = LinearForm(lin).assemble(ub, z=z)
    assert np.abs(a - make(CLinearForm, lin).assemble(ub, z=z)).max() < 1e-12
