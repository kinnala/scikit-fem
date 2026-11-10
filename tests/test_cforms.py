import shutil
import warnings

import numpy as np
import pytest

from skfem import (BilinearForm, LinearForm, Basis, FacetBasis, MeshTri,
                   MeshTet, MeshQuad, MeshHex, MeshLine, ElementTriP2,
                   ElementTetP1, ElementQuad2, ElementTriMini, ElementDG,
                   ElementTetP2, ElementLineP2, ElementVector, ElementTriP1,
                   ElementHex1, MeshTri2, MeshQuad2, ElementLineP1,
                   ElementTriRT0, ElementTriRT1, ElementTriRT2, ElementTriBDM1,
                   ElementTetRT1, ElementHexRT1, ElementTriN3, ElementQuadN1,
                   ElementTetN0, ElementTetN1, ElementTriP3, ElementTriP4,
                   ElementTriP1DG, ElementQuad1)
from skfem.helpers import (dot, grad, inner, ddot, sym_grad, div, curl,
                           mul)
from skfem.models.elasticity import linear_stress

pytestmark = [
    pytest.mark.skipif(shutil.which('cc') is None, reason="No C compiler"),
    # an unexpected fallback to Python would hide errors in the C path
    pytest.mark.filterwarnings("error:.*falls back to Python"),
]

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


@pytest.mark.parametrize("nthreads", [2, 3, 4, 8, 16])
@pytest.mark.parametrize("make", [
    lambda: Basis(MeshTet().refined(3), ElementTetP2()),
    lambda: Basis(MeshTri().refined(5), ElementVector(ElementTriP1()),
                  elements=lambda x: x[0] < .7),
    lambda: Basis(MeshQuad().refined(5), ElementDG(ElementQuad2())),
    lambda: Basis(MeshHex().refined(3), ElementVector(ElementHex1())),
])
def test_partition_disjoint_rows(nthreads, make):
    from skfem.experimental.cforms import _partition
    basis = make()
    perm, parts, phases = _partition(basis, nthreads)
    assert np.array_equal(np.sort(perm), np.arange(basis.nelems))
    assert parts[0] == 0 and parts[-1] == basis.nelems
    assert np.all(np.diff(parts) >= 0)
    assert phases[0] == 0 and phases[-1] == len(parts) - 1
    for f in range(len(phases) - 1):
        groups = range(phases[f], phases[f + 1])
        assert len(groups) <= nthreads
        rows = [np.unique(basis.element_dofs[:, perm[parts[g]:parts[g + 1]]])
                for g in groups]
        allrows = np.concatenate(rows)
        assert len(allrows) == len(np.unique(allrows))  # disjoint
    if nthreads <= 8:  # with tiny blocks most elements are on interfaces
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


@pytest.mark.parametrize("m,e,in_c", [
    (MeshTri().refined(3), ElementTriP2(), True),            # affine
    (MeshTet().refined(2), ElementTetP1(), True),
    (MeshHex().refined(2), ElementHex1(), True),             # isoparametric
    (MeshTri2.init_circle(2), ElementTriP2(), True),         # curved
    (MeshQuad2().refined(2).translated((0., 1.)), ElementQuad2(), True),
    (MeshLine().refined(3), ElementLineP1(), True),
])
@pytest.mark.parametrize("subset", [False, True])
def test_global_coordinates(m, e, in_c, subset):
    basis = Basis(m, e, elements=(lambda x: x[0] < .5) if subset else None)

    def bilin(u, v, w):
        return (np.exp(w.x[0]) + w.x[-1] ** 2 * w.h) * u * v

    def lin(v, w):
        return np.sin(w.x[0]) * v

    form = CBilinearForm(bilin)
    A = BilinearForm(bilin).assemble(basis)
    assert abs(A - form.assemble(basis)).max() < 1e-12
    a = LinearForm(lin).assemble(basis)
    assert np.abs(a - CLinearForm(lin).assemble(basis)).max() < 1e-12
    # 'x' is computed inside the kernel instead of passed
    assert ("XQ[g] +=" in form.source(basis)) == in_c

    # user-given 'x' overrides the default
    x = basis.interpolate(basis.zeros() + 1.)
    A = BilinearForm(bilin).assemble(basis, x=x)
    assert abs(A - form.assemble(basis, x=x)).max() < 1e-12


@pytest.mark.parametrize("cache_sparsity", [False, True])
def test_no_compiler(monkeypatch, cache_sparsity):
    import skfem.experimental.cforms as cforms
    # e.g. Pyodide, where sys.platform == 'emscripten'
    monkeypatch.setattr(cforms.sys, 'platform', 'emscripten')
    basis = Basis(MeshTri().refined(2), ElementTriP1())
    form = CBilinearForm(laplace, cache_sparsity=cache_sparsity)
    with pytest.warns(UserWarning, match="no C compiler"):
        B = form.assemble(basis)
    assert abs(BilinearForm(laplace).assemble(basis) - B).max() < 1e-12
    with pytest.warns(UserWarning, match="no C compiler"):
        b = CLinearForm(load).assemble(basis, u0=basis.zeros(), k=1.)
    assert np.abs(LinearForm(load).assemble(basis, u0=basis.zeros(), k=1.)
                  - b).max() < 1e-12
    monkeypatch.setattr(cforms.sys, 'platform', 'linux')
    monkeypatch.setenv('CC', 'no-such-compiler')
    with pytest.warns(UserWarning, match="no C compiler"):
        form.assemble(basis)


def _shuffled(m, seed=0):
    """The same mesh with randomly permuted local vertex orders; elements
    then have facets and edges in both directions."""
    rng = np.random.default_rng(seed)
    t = m.t.copy()
    for k in range(t.shape[1]):
        t[:, k] = t[rng.permutation(t.shape[0]), k]
    return type(m)(m.p, t)


def _hdiv(u, v, w):
    return dot(u, v) + div(u) * div(v)


def _hcurl2(u, v, w):
    return dot(u, v) + curl(u) * curl(v)


def _hcurl3(u, v, w):
    return dot(u, v) + dot(curl(u), curl(v))


def _distorted(m, seed=0):
    rng = np.random.default_rng(seed)
    interior = (m.p * (1 - m.p) > 0).all(axis=0)
    return type(m)(m.p + .03 * rng.standard_normal(m.p.shape) * interior,
                   m.t)


@pytest.mark.parametrize("make,form", [
    (lambda: Basis(_shuffled(MeshTri().refined(2)), ElementTriRT0()), _hdiv),
    (lambda: Basis(_shuffled(MeshTri().refined(2)), ElementTriRT2()), _hdiv),
    (lambda: Basis(_shuffled(MeshTri().refined(2)), ElementTriBDM1()),
     _hdiv),
    (lambda: Basis(_shuffled(MeshTet().refined(1)), ElementTetRT1()), _hdiv),
    (lambda: Basis(_distorted(MeshHex().refined(2)), ElementHexRT1()),
     _hdiv),
    (lambda: Basis(_shuffled(MeshTri().refined(2)), ElementTriN3()),
     _hcurl2),
    (lambda: Basis(_distorted(MeshQuad().refined(3)), ElementQuadN1()),
     _hcurl2),
    (lambda: Basis(_shuffled(MeshTet().refined(1)), ElementTetN1()),
     _hcurl3),
    (lambda: Basis(MeshTri2.init_circle(2), ElementTriRT1()), _hdiv),
    (lambda: Basis(_shuffled(MeshTri().refined(2)), ElementTriP4()),
     laplace),
    (lambda: Basis(_shuffled(MeshTri().refined(2)),
                   ElementVector(ElementTriP3())), elasticity),
])
@pytest.mark.parametrize("cache_sparsity", [False, True])
def test_more_elements(make, form, cache_sparsity):
    basis = make()
    A = BilinearForm(form).assemble(basis)
    B = CBilinearForm(form, cache_sparsity=cache_sparsity).assemble(basis)
    assert abs(A - B).max() < 1e-12 * abs(A).max()


def _stokes(u, p, v, q, w):
    return ddot(grad(u), grad(v)) - div(v) * p - div(u) * q + p * q


def _mixed(s, u, t, v, w):
    return dot(s, t) + div(s) * v + div(t) * u


def _maxwell(E, p, F, q, w):
    return dot(curl(E), curl(F)) + dot(E, grad(q)) + dot(F, grad(p))


@pytest.mark.parametrize("make,form", [
    (lambda: Basis(_shuffled(MeshTri().refined(2)),
                   ElementVector(ElementTriP3()) * ElementTriP2()), _stokes),
    (lambda: Basis(_shuffled(MeshTri().refined(2)),
                   ElementTriRT1() * ElementDG(ElementTriP1())), _mixed),
    (lambda: Basis(_shuffled(MeshTet().refined(1)),
                   ElementTetN0() * ElementTetP1()), _maxwell),
    (lambda: Basis(_distorted(MeshQuad().refined(3)),
                   ElementVector(ElementQuad2()) * ElementQuad1(),
                   elements=lambda x: x[0] < .6), _stokes),
])
@pytest.mark.parametrize("nthreads", [1, 3])
def test_composite(make, form, nthreads):
    basis = make()
    A = BilinearForm(form).assemble(basis)
    for cache_sparsity in [False, True]:
        B = (CBilinearForm(form, nthreads=nthreads,
                           cache_sparsity=cache_sparsity)
             .assemble(basis))
        assert abs(A - B).max() < 1e-12 * abs(A).max()


@pytest.mark.parametrize("interpolate_in_c", [False, True])
def test_composite_coefficients(interpolate_in_c):
    rng = np.random.default_rng(0)
    basis = Basis(_shuffled(MeshTri().refined(2)),
                  ElementVector(ElementTriP3()) * ElementTriP2())
    z = rng.standard_normal(basis.N)

    def jac(u, p, v, q, w):  # Navier-Stokes
        return (ddot(grad(u), grad(v))
                + dot(mul(grad(u), w.z[0]) + mul(grad(w.z[0]), u), v)
                - div(v) * p - div(u) * q + w.z[1] * p * q)

    def res(v, q, w):
        return dot(mul(grad(w.z[0]), w.z[0]), v) + w.z[1] * q

    a, b = CBilinearForm(jac), CLinearForm(res)
    a.interpolate_in_c = b.interpolate_in_c = interpolate_in_c
    A = BilinearForm(jac).assemble(basis, z=z)
    assert abs(A - a.assemble(basis, z=z)).max() < 1e-12 * abs(A).max()
    f = LinearForm(res).assemble(basis, z=z)
    assert np.abs(f - b.assemble(basis, z=z)).max() < 1e-12 * np.abs(f).max()


@pytest.mark.parametrize("make,form,linear", [
    (lambda: Basis(_shuffled(MeshTri().refined(2)), ElementTriRT1()),
     lambda u, v, w: (1. + dot(w.c, w.c) + w.c.div ** 2) * dot(u, v),
     False),
    (lambda: Basis(_shuffled(MeshTet().refined(1)), ElementTetN1()),
     lambda v, w: dot(curl(w.c), curl(v)) + dot(w.c, v), True),
    (lambda: Basis(_shuffled(MeshTri().refined(2)), ElementTriP3()),
     lambda u, v, w: ((1. + w.c ** 2) * dot(grad(u), grad(v))
                      + dot(grad(w.c), grad(v)) * u), False),
])
def test_coefficients_more_elements(make, form, linear):
    basis = make()
    c = np.random.default_rng(0).standard_normal(basis.N)
    ref, cls = (LinearForm, CLinearForm) if linear else (BilinearForm,
                                                         CBilinearForm)
    A = ref(form).assemble(basis, c=c)
    B = cls(form).assemble(basis, c=c)
    assert np.abs(A - B).max() < 1e-12 * np.abs(A).max()


def test_mixed_bases_hdiv():
    ub = Basis(_shuffled(MeshTri().refined(2)), ElementTriRT1())
    vb = ub.with_element(ElementTriP1DG())

    def b(u, v, w):
        return div(u) * v

    A = BilinearForm(b).assemble(ub, vb)
    for cache_sparsity in [False, True]:
        B = CBilinearForm(b, cache_sparsity=cache_sparsity).assemble(ub, vb)
        assert abs(A - B).max() < 1e-12


def test_more_functions():
    basis = Basis(MeshTri().refined(2), ElementTriP2())

    def form(u, v, w):
        x, y = w.x
        return (np.arctan2(y, 1. + x) + np.hypot(x, y) + np.log1p(x)
                + np.expm1(y) + np.cbrt(1. + x) + np.arcsinh(y)
                + np.log2(2. + x) + np.log10(2. + y) + np.exp2(x)
                + np.arctanh(.5 * y) + np.arccosh(2. + x)) * u * v

    A = BilinearForm(form).assemble(basis)
    assert abs(A - CBilinearForm(form).assemble(basis)).max() < 1e-12


def test_component_pairs_skipped():
    basis = Basis(MeshTri().refined(2), ElementVector(ElementTriP1()))
    form = CBilinearForm(lambda u, v, w: ddot(grad(u), grad(v)))
    # u and v in different components do not interact
    assert form.source(basis).count("A[IDX_U") == 2
    form = CBilinearForm(elasticity)
    assert form.source(basis).count("A[IDX_U") == 4


def test_trace_cache():
    basis = Basis(MeshTri().refined(2), ElementTriP2())
    u0 = np.sin(basis.doflocs[0])

    def form(u, v, w):
        return (w.k + w.u0 ** 2) * dot(grad(u), grad(v))

    cform = CBilinearForm(form)
    for k, c in [(1., 1.), (2., 3.)]:  # new values, same structure
        A = BilinearForm(form).assemble(basis, k=k, u0=c * u0)
        assert abs(A - cform.assemble(basis, k=k, u0=c * u0)).max() < 1e-12
    assert len(cform._trace_cache) == 1
    # different structure of 'w' is traced again
    u0 = basis.interpolate(u0)
    A = BilinearForm(form).assemble(basis, k=1., u0=u0)
    assert abs(A - cform.assemble(basis, k=1., u0=u0)).max() < 1e-12
    assert len(cform._trace_cache) == 2
