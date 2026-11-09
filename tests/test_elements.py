from unittest import TestCase, main

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pytest

from skfem.element import (
    ElementHex1,
    ElementHexS2,
    ElementLineP0,
    ElementLineP1,
    ElementLineP2,
    ElementLinePp,
    ElementLineMini,
    ElementQuad0,
    ElementQuad1,
    ElementQuad2,
    ElementQuadP,
    ElementQuadRT0,
    ElementQuadS2,
    ElementTetMini,
    ElementTetP0,
    ElementTetP1,
    ElementTetP2,
    ElementTriMini,
    ElementTriP0,
    ElementTriP1,
    ElementTriP2,
    ElementTriP3,
    ElementTriP4,
    ElementTriP,
    ElementTriRT,
    ElementTriRT1,
    ElementTriRT2,
    ElementTriBDM1,
    ElementTriN1,
    ElementTriN2,
    ElementTriN3,
    ElementTriHHJ1,
    ElementTriRT0,
    ElementVectorH1,
    ElementHex2,
    ElementQuadBFS,
    ElementTriCR,
    ElementTriCCR,
    ElementTriP1B,
    ElementTriP2B,
    ElementTetCR,
    ElementTetCCR,
    ElementTriHermite,
    ElementTriMorley,
    ElementTriArgyris,
    ElementTriDG,
    ElementTetDG,
    ElementQuadDG,
    ElementQuadP,
    ElementHexDG,
    ElementWedge1,
    ElementQuadDPC,
    ElementHexDPC,
    ElementLineBubble,
    ElementTriBubble,
    ElementTetBubble,
    ElementTetMorley,
    ElementTetMWX3,
)
from skfem.mesh import MeshHex, MeshLine, MeshQuad, MeshTet, MeshTri
from skfem.assembly import (Basis, FacetBasis, Functional, BilinearForm,
                            LinearForm, InteriorFacetBasis)
from skfem.helpers import dot, grad, div
from skfem.models.poisson import laplace, unit_load
from skfem.utils import solve, condense
from skfem.mapping import MappingAffine
from skfem.refdom import RefTet


class TestNodality(TestCase):
    """Test for Element.doflocs."""

    elems = [
        ElementLineP0(),
        ElementLineP1(),
        ElementLineP2(),
        ElementLinePp(1),
        ElementLinePp(3),
        ElementLineMini(),
        ElementTriP0(),
        ElementTriP1(),
        ElementTriP2(),
        ElementTriP3(),
        ElementTriP4(),
        ElementTriMini(),
        ElementQuad0(),
        ElementQuad1(),
        ElementQuad2(),
        ElementQuadS2(),
        ElementQuadP(1),
        ElementQuadP(3),
        ElementTetP0(),
        ElementTetP1(),
        ElementTetP2(),
        ElementTetMini(),
        ElementHex1(),
        ElementHexS2(),
        ElementHex2(),
        ElementTetCR(),
        ElementTetCCR(),
        ElementTriCR(),
        ElementTriCCR(),
        ElementWedge1(),
        ElementLineBubble(),
        ElementTriBubble(),
        ElementTetBubble(),
        ElementTriP(1),
        ElementTriP(5),
    ]

    def runTest(self):
        for e in self.elems:
            N = e.doflocs.shape[0]
            Ih = np.zeros((N, N))
            for itr in range(N):
                Ih[itr] = e.lbasis(e.doflocs.T, itr)[0]

            # Remove nan-rows: test nodality only on non-nan doflocs.
            #
            # Some elements, such as ElementTriMini might have a combination
            # of nodal dofs and non-nodal dofs.
            #
            # Nodal dof is defined so that there exists a point where the
            # corresponding basis function is one, and other basis functions
            # are zero. Non-nodal dof does not satisfy this property.
            ix = np.isnan(np.sum(Ih, axis=1))
            Nnan = np.sum(ix)
            ixs = np.nonzero(~ix)[0]
            Ih = Ih[ixs].T[ixs].T

            assert_allclose(Ih, np.eye(N - Nnan), atol=1e-13,
                            err_msg="{}".format(type(e)))


class TestNodalityTriRT0(TestCase):

    elem = ElementTriRT0()

    def runTest(self):
        e = self.elem
        N = e.doflocs.shape[0]
        Ih = np.zeros((N, N))
        normals = np.array([[0., -1.],
                            [1 / np.sqrt(2), 1 / np.sqrt(2)],
                            [-1., 0.]]).T
        for itr in range(N):
            # calculate integral of normal component over edge
            A = np.sum(e.lbasis(e.doflocs.T, itr)[0] * normals, axis=0)
            n = np.array([1., np.sqrt(2), 1.])
            Ih[itr] = A * n

        assert_allclose(Ih, np.eye(N),
                        err_msg="{}".format(type(e)))


class TestNodalityQuadRT0(TestCase):

    elem = ElementQuadRT0()

    def runTest(self):
        e = self.elem
        N = e.doflocs.shape[0]
        Ih = np.zeros((N, N))
        normals = np.array([[0., -1.],
                            [1., 0.],
                            [0., 1.],
                            [-1., 0.]]).T
        for itr in range(N):
            # calculate integral of normal component over edge
            A = np.sum(e.lbasis(e.doflocs.T, itr)[0] * normals, axis=0)
            n = np.ones(4)
            Ih[itr] = A * n

        assert_allclose(Ih, np.eye(N),
                        err_msg="{}".format(type(e)))


class TestComposite(TestCase):

    def runTest(self):
        from skfem.element.element_composite import ElementComposite

        self.check_equivalence(
            ElementComposite(ElementTriP1(),
                             ElementTriP1()),
            ElementVectorH1(ElementTriP1())
        )

    def check_equivalence(self, ec, ev):
        X = np.array([[0.125, 0.1111], [0.0555, 0.6]])
        m = MeshTri.init_refdom()
        mapping = MappingAffine(m)

        for k in range(6):
            for i in [0, 1]:
                # accessing i'th component looks slightly different
                assert_array_equal(
                    ev.gbasis(mapping, X, k)[0].value[i],
                    ec.gbasis(mapping, X, k)[i].value
                )
                for j in [0, 1]:
                    assert_array_equal(
                        ev.gbasis(mapping, X, k)[0].grad[i][j],
                        ec.gbasis(mapping, X, k)[i].grad[j]
                    )


class TestCompositeMul(TestComposite):

    def runTest(self):

        self.check_equivalence(
            ElementTriP1() * ElementTriP1(),
            ElementVectorH1(ElementTriP1())
        )


class TestCompatibilityWarning(TestCase):

    meshes = [
        MeshTet,
        MeshQuad,
        MeshHex,
        MeshLine,
    ]
    elem = ElementTriP1

    def runTest(self):

        for m in self.meshes:

            def init_incompatible():
                return Basis(m(), self.elem())

            self.assertRaises(ValueError, init_incompatible)


class TestDerivatives(TestCase):
    """Test values of derivatives."""

    elems = [
        ElementLineP0(),
        ElementLineP1(),
        ElementLineP2(),
        ElementLineMini(),
        ElementTriP0(),
        ElementTriP1(),
        ElementTriP2(),
        ElementTriP3(),
        ElementTriP4(),
        ElementTriMini(),
        ElementQuad0(),
        ElementQuad1(),
        ElementQuad2(),
        ElementQuadS2(),
        ElementTetP0(),
        ElementTetP1(),
        ElementTetP2(),
        ElementTetMini(),
        ElementHex1(),
        ElementHexS2(),
        ElementHex2(),
        ElementTriCR(),
        ElementTriCCR(),
        ElementTetCR(),
        ElementTetCCR(),
        ElementWedge1(),
        ElementQuadDPC(0),
        ElementQuadDPC(3),
        ElementHexDPC(2),
        ElementLineBubble(),
        ElementTriBubble(),
        ElementTetBubble(),
        ElementTriP(2),
        ElementTriP(6),
    ]

    def runTest(self):
        for elem in self.elems:
            eps = 1e-6
            for base in [0., .3, .6, .9]:
                if elem.dim == 1:
                    y = np.array([[base, base + eps]])
                elif elem.dim == 2:
                    y = np.array([[base, base + eps, base, base],
                                  [base, base, base, base + eps]])
                elif elem.dim == 3:
                    y = np.array([[base, base + eps, base, base, base, base],
                                  [base, base, base, base + eps, base, base],
                                  [base, base, base, base, base, base + eps]])
                i = 0
                while True:
                    try:
                        out = elem.lbasis(y, i)
                    except ValueError:
                        break
                    diff = (out[0][1] - out[0][0]) / eps
                    errmsg = 'x-derivative for {}th bfun failed for {}'
                    self.assertAlmostEqual(diff, out[1][0][0], delta=1e-3,
                                           msg=errmsg.format(i, elem))
                    if elem.dim > 1:
                        diff = (out[0][3] - out[0][2]) / eps
                        errmsg = 'y-derivative for {}th bfun failed for {}'
                        self.assertAlmostEqual(diff, out[1][1][3], delta=1e-3,
                                               msg=errmsg.format(i, elem))
                    if elem.dim == 3:
                        diff = (out[0][5] - out[0][4]) / eps
                        errmsg = 'z-derivative for {}th bfun failed for {}'
                        self.assertAlmostEqual(diff, out[1][2][4], delta=1e-3,
                                               msg=errmsg.format(i, elem))
                    i += 1


class TestPartitionofUnity(TestCase):
    """Test that elements form a partition of unity."""

    elems = [
        ElementLineP1(),
        ElementLineP2(),
        ElementTriP1(),
        ElementTriP2(),
        ElementTriP3(),
        ElementTriP4(),
        ElementQuad1(),
        ElementQuad2(),
        ElementQuadS2(),
        ElementTetP1(),
        ElementTetP2(),
        ElementHex1(),
        ElementHexS2(),
        ElementHex2(),
        ElementTetCR(),
        ElementTetCCR(),
        ElementTriCR(),
        ElementTriP1B(),
        ElementTriP2B(),
        ElementWedge1(),
        ElementTriP(5),
    ]

    def runTest(self):
        for elem in self.elems:
            if elem.dim == 1:
                y = np.array([[.15]])
            elif elem.dim == 2:
                y = np.array([[.15],
                              [.15]])
            elif elem.dim == 3:
                y = np.array([[.15],
                              [.15],
                              [.15]])
            out = 0.
            for i in range(elem.doflocs.shape[0]):
                if np.isnan(elem.doflocs[i]).any():
                    continue
                out += elem.lbasis(y, i)[0][0]
            self.assertAlmostEqual(out, 1, msg='failed for {}'.format(elem))


class TestElementLinePp(TestCase):

    def test_p_less_than_1_error(self):
        """Tests that exception is thrown when initializing with p < 1."""
        with self.assertRaises(ValueError):
            ElementLinePp(0)


class TestElementQuadBFS(TestCase):

    def test_throw_index_error(self):
        """Tests that exception is thrown when i % 4 not in (0, 1, 2, 3)."""
        element = ElementQuadBFS()
        with self.assertRaises(ValueError):
            element.gdof(0, 0, -1)
        with self.assertRaises(ValueError):
            element.gdof(0, 0, 16)


@pytest.mark.parametrize(
    "m,e,edg",
    [
        (MeshTri().refined(), ElementTriP1(), ElementTriDG),
        (MeshTri().refined(), ElementTriP2(), ElementTriDG),
        (MeshTet().refined(), ElementTetP1(), ElementTetDG),
        (MeshTet().refined(), ElementTetP2(), ElementTetDG),
        (MeshTri().refined(), ElementTriArgyris(), ElementTriDG),
        (MeshTri().refined(), ElementTriMorley(), ElementTriDG),
        (MeshTri().refined(), ElementTriHermite(), ElementTriDG),
        (MeshHex().refined(), ElementHex1(), ElementHexDG),
        (MeshQuad().refined(), ElementQuad1(), ElementQuadDG),
    ]
)
def test_dg_element(m, e, edg):

    edg = edg(e)

    @Functional
    def square(w):
        return w['random'] ** 2

    basis = Basis(m, e)
    basisdg = Basis(m, edg)

    assert_allclose(
        square.assemble(
            basis,
            random=basis.interpolate(
                basis.zeros() + 1)),
        square.assemble(
                basisdg,
                random=basisdg.interpolate(
                    basisdg.zeros() + 1)),
                     )


@pytest.mark.parametrize(
    "e,edg",
    [
        (ElementTriP1(), ElementTriDG),
        (ElementTetP2(), ElementTetDG),
        (ElementTriArgyris(), ElementTriDG),
        (ElementQuad1(), ElementQuadDG),
        (ElementQuadP(4), ElementQuadDG),
        (ElementHex2(), ElementHexDG),
    ]
)
def test_initialize_dg_composite_elements(e, edg):
    E = edg(e) * e


@pytest.mark.parametrize(
    "m,e,ndofs",
    [
        (MeshQuad(), ElementQuadDPC(0), 1),
        (MeshQuad(), ElementQuadDPC(1), 3),
        (MeshQuad(), ElementQuadDPC(3), 10),
        (MeshHex(), ElementHexDPC(0), 1),
        (MeshHex(), ElementHexDPC(1), 4),
        (MeshHex(), ElementHexDPC(3), 20),
    ]
)
def test_dpc_orthonormal(m, e, ndofs):
    """The basis is orthonormal over the reference element."""
    basis = Basis(m, e)
    assert basis.N == ndofs

    @BilinearForm
    def mass(u, v, w):
        return u * v

    assert_allclose(mass.assemble(basis).toarray(), np.eye(ndofs),
                    atol=1e-12)


@pytest.mark.parametrize(
    "m,e",
    [
        (MeshQuad().refined(1), ElementQuadDPC(1)),
        (MeshQuad().refined(1), ElementQuadDPC(2)),
        (MeshHex().refined(1), ElementHexDPC(1)),
        (MeshHex().refined(1), ElementHexDPC(2)),
    ]
)
def test_dpc_polynomial_reproduction(m, e):
    """Total degree p is reproduced exactly while the degree p tensor
    product monomial is not."""
    basis = Basis(m, e)
    dim = m.dim()
    p = e.p

    def error(exps):
        def fun(x):
            return np.prod([x[i] ** exps[i] for i in range(dim)], axis=0)
        y = basis.project(fun)
        X = np.asarray(basis.global_coordinates())
        return np.max(np.abs(np.asarray(basis.interpolate(y)) - fun(X)))

    for exps in np.ndindex(*(dim * (p + 1,))):
        if sum(exps) <= p:
            assert error(exps) < 1e-12, exps
    assert error(dim * (p,)) > 1e-3

@pytest.mark.parametrize(
    "m,e",
    [
        (MeshLine().refined(2), ElementLineBubble()),
        (MeshTri().refined(2), ElementTriBubble()),
        (MeshTet().refined(1), ElementTetBubble()),
    ]
)
def test_bubble_vanishes_on_facets(m, e):
    basis = Basis(m, e)
    assert basis.N == m.nelements
    assert len(basis.get_dofs()) == 0
    fbasis = FacetBasis(m, e, facets=np.arange(m.nfacets))
    for i in range(e.interior_dofs):
        assert_allclose(fbasis.basis[i][0], 0., atol=1e-13)


@pytest.mark.parametrize(
    "m,e1,eb,emini",
    [
        (MeshLine().refined(3), ElementLineP1(), ElementLineBubble(),
         ElementLineMini()),
        (MeshTri().refined(3), ElementTriP1(), ElementTriBubble(),
         ElementTriMini()),
        (MeshTet().refined(2), ElementTetP1(), ElementTetBubble(),
         ElementTetMini()),
    ]
)
def test_bubble_enrichment_equals_mini(m, e1, eb, emini):
    """Composite P1 * bubble spans the same space as the MINI element."""

    @BilinearForm
    def bilinf(u, ub, v, vb, w):
        return dot(grad(u) + grad(ub), grad(v) + grad(vb))

    @LinearForm
    def linf(v, vb, w):
        return v + vb

    basis = Basis(m, e1 * eb, intorder=2 * emini.maxdeg)
    x = solve(*condense(bilinf.assemble(basis),
                        linf.assemble(basis),
                        D=basis.get_dofs()))
    (x1, _), (xb, _) = basis.split(x)

    mbasis = Basis(m, emini)
    y = solve(*condense(laplace.assemble(mbasis),
                        unit_load.assemble(mbasis),
                        D=mbasis.get_dofs()))

    assert_allclose(x1, y[mbasis.nodal_dofs[0]], atol=1e-12)
    assert_allclose(xb, y[mbasis.interior_dofs[0]], atol=1e-12)


def _distorted_tet():
    m = MeshTet().refined(1)
    rng = np.random.default_rng(0)
    p = m.p + .2 * m.param() * (rng.random(m.p.shape) - .5)
    A = np.array([[1., .3, .2],
                  [0., 1., .4],
                  [0., 0., 1.]])
    return MeshTet(A @ p, m.t)


@pytest.mark.parametrize("e", [ElementTetMorley(), ElementTetMWX3()])
def test_mwx_dofs(e):
    """Evaluate the DOF functionals of a random function in each element and
    compare to the global DOF values.  This verifies the DOF definitions and
    the orientation of the normals."""
    m = _distorted_tet()

    # normals defined using the global vertex indices, sorted in mesh.edges
    # and mesh.facets
    a, b = m.p[:, m.edges[0]], m.p[:, m.edges[1]]
    t = (b - a) / np.linalg.norm(b - a, axis=0)
    ax = np.zeros_like(t)
    ax[np.argmin(np.abs(t), axis=0), np.arange(t.shape[1])] = 1.
    en1 = np.cross(t, ax, axis=0)
    en1 /= np.linalg.norm(en1, axis=0)
    en2 = np.cross(t, en1, axis=0)
    a, b, c = (m.p[:, m.facets[i]] for i in range(3))
    fn = np.cross(b - a, c - a, axis=0)
    fn /= np.linalg.norm(fn, axis=0)

    # evaluate at vertices, edge (start, midpoint, end), face centroids
    P = RefTet.p
    X = [P[:, i] for i in range(4)]
    for i, j in RefTet.edges:
        X += [P[:, i], .5 * (P[:, i] + P[:, j]), P[:, j]]
    X += [P[:, f].mean(axis=1) for f in RefTet.facets]
    X = np.array(X).T
    basis = Basis(m, e, quadrature=(X, np.ones(X.shape[1])))
    x = np.random.default_rng(1).random(basis.N)
    u = basis.interpolate(x)
    val, grad, hess = np.asarray(u), np.asarray(u.grad), np.asarray(u.hess)

    def edge_mean(f, k):
        j = 4 + 3 * k
        return (f[..., j] + 4. * f[..., j + 1] + f[..., j + 2]) / 6.

    dofs = []
    if e.nodal_dofs > 0:
        dofs += [val[:, i] for i in range(4)]
    for k in range(6):
        if e.edge_dofs == 1:
            dofs += [edge_mean(val, k)]
        else:
            g = edge_mean(grad, k)
            dofs += [np.einsum('ik,ik->k', g, en1[:, m.t2e[k]]),
                     np.einsum('ik,ik->k', g, en2[:, m.t2e[k]])]
    for k in range(4):
        n = fn[:, m.t2f[k]]
        if e.maxdeg == 2:
            dofs += [np.einsum('ik,ik->k', grad[..., 22 + k], n)]
        else:
            dofs += [np.einsum('ijk,ik,jk->k', hess[..., 22 + k], n, n)]

    assert_allclose(np.array(dofs), x[basis.element_dofs], atol=1e-11)


@pytest.mark.parametrize(
    "e,maxdeg",
    [
        (ElementTetMorley(), 2),
        (ElementTetMWX3(), 3),
    ]
)
def test_mwx_polynomial_reproduction(e, maxdeg):
    """Polynomials of degree k are reproduced, degree k + 1 are not."""
    basis = Basis(_distorted_tet(), e)
    X = np.asarray(basis.global_coordinates())

    @BilinearForm
    def mass(u, v, w):
        return u * v

    @LinearForm
    def load(v, w):
        return w['f'] * v

    M = mass.assemble(basis)

    def error(exps):
        f = np.prod([X[i] ** exps[i] for i in range(3)], axis=0)
        y = solve(M, load.assemble(basis, f=f))
        return np.max(np.abs(np.asarray(basis.interpolate(y)) - f))

    for exps in np.ndindex(*(3 * (maxdeg + 1,))):
        if sum(exps) <= maxdeg:
            assert error(exps) < 1e-10, exps
    assert error((maxdeg + 1, 0, 0)) > 1e-4


def _gbasis_reference(e, mapping, X, i):
    """Unoptimized evaluation of ElementGlobal basis function."""
    import itertools
    x = mapping.F(X)
    h = e._h[:, None]
    y = (x - e._c[:, :, None]) / h
    V = e.V
    U = [np.zeros((e.dim,) * k + x[0].shape)
         for k in range(e.derivatives + 1)]
    for k in range(e.derivatives + 1):
        for diff in itertools.product(*((list(range(e.dim)),) * k)):
            for itr, p in enumerate(e._pbasis[diff]):
                U[k][diff] += V[:, itr, i][:, None] * p(*y) / h ** k
    return U


@pytest.mark.parametrize(
    "m,e",
    [
        (MeshTri().refined(2), ElementTriArgyris()),
        (MeshQuad().refined(2), ElementQuadBFS()),
        (_distorted_tet(), ElementTetMorley()),
        (_distorted_tet(), ElementTetMWX3()),
    ]
)
def test_element_global_gbasis(m, e):
    """Compare the vectorized and cached ElementGlobal.gbasis to the
    unoptimized loop, also when the calls are interleaved."""
    mapping = m.mapping()
    X1 = Basis(m, e).X
    X2 = X1[:, ::-1].copy()
    N = len(e.doflocs)
    for i in list(range(N - 1, -1, -2)) + list(range(0, N, 2)):
        for X in [X1, X2]:
            U = _gbasis_reference(e, mapping, X, i)
            field = e.gbasis(mapping, X, i)[0]
            fields = [np.asarray(field), field.grad, field.hess]
            fields += [getattr(field, 'grad{}'.format(k + 3))
                       for k in range(e.derivatives - 2)]
            for u, v in zip(U, fields):
                assert_allclose(v, u, rtol=1e-8,
                                atol=1e-8 * np.abs(u).max())


@pytest.mark.parametrize(
    "m1,m2,e",
    [
        (MeshTri().refined(2), MeshTri().refined(2).scaled(2.),
         ElementTriMorley),
        (MeshTet().refined(1), _distorted_tet(), ElementTetMorley),
    ]
)
def test_element_global_reuse(m1, m2, e):
    """The same instance of ElementGlobal can be used with several
    meshes."""
    elem = e()
    Basis(m1, elem)
    basis = Basis(m2, elem)
    basis_fresh = Basis(m2, e())
    for f, g in zip(basis.basis, basis_fresh.basis):
        assert_allclose(f[0].hess, g[0].hess)


@pytest.mark.parametrize(
    "m,e,fun",
    [
        (MeshTri().refined(5), ElementTriArgyris(),
         lambda x: x[0] ** 5 - 3. * x[0] ** 2 * x[1] ** 3 + x[1]),
        (MeshQuad().refined(5), ElementQuadBFS(),
         lambda x: x[0] ** 3 * x[1] ** 3 - x[0] * x[1] ** 2),
    ]
)
def test_element_global_conditioning(m, e, fun):
    """Polynomials are reproduced accurately also on fine meshes."""
    basis = Basis(m, e)
    y = basis.project(fun)
    X = np.asarray(basis.global_coordinates())
    assert_allclose(np.asarray(basis.interpolate(y)), fun(X), atol=1e-11)


def _unsorted_tri():
    """A triangle mesh where the element vertices are not sorted."""
    m = MeshTri().refined(2)
    rot = np.random.default_rng(1).integers(0, 3, m.t.shape[1])
    t = np.array([np.roll(m.t[:, k], r) for k, r in enumerate(rot)]).T
    return MeshTri(m.p, t, sort_t=False)


@pytest.mark.parametrize(
    "m",
    [
        MeshTri().refined(2),
        _unsorted_tri(),
    ]
)
@pytest.mark.parametrize(
    "e",
    [
        ElementTriP3(),
        ElementTriP4(),
        ElementTriP(3),
        ElementTriP(5),
        ElementTriP(6),
    ]
)
def test_tri_facet_dof_orientation(m, e):
    """The basis is continuous also if the element vertices are not
    sorted."""
    fbasis = [InteriorFacetBasis(m, e, side=i) for i in range(2)]
    x = np.random.default_rng(0).random(fbasis[0].N)
    assert_allclose(np.asarray(fbasis[0].interpolate(x)),
                    np.asarray(fbasis[1].interpolate(x)),
                    atol=1e-12)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 7])
def test_tri_p_hierarchical(p):
    """The basis of degree p is contained in the basis of degree p + 1."""
    e1, e2 = ElementTriP(p), ElementTriP(p + 1)
    assert (e1.doflocs.shape[0]
            == 3 + 3 * e1.facet_dofs + e1.interior_dofs
            == (p + 1) * (p + 2) // 2)
    X = np.random.default_rng(0).random((2, 10)) / 2.
    for i in range(e1.doflocs.shape[0]):
        if i < 3:
            j = i
        elif i < 3 + 3 * e1.facet_dofs:
            facet, k = divmod(i - 3, e1.facet_dofs)
            j = 3 + facet * e2.facet_dofs + k
        else:
            ix = e1._interior[i - 3 - 3 * e1.facet_dofs]
            j = 3 + 3 * e2.facet_dofs + e2._interior.index(ix)
        for f1, f2 in zip(e1.lbasis(X, i), e2.lbasis(X, j)):
            assert_allclose(f1, f2, atol=1e-14)


@pytest.mark.parametrize(
    "p,elem",
    [
        (3, ElementTriP3),
        (4, ElementTriP4),
    ]
)
def test_tri_p_same_space(p, elem):
    """The hierarchical and the nodal bases give the same solution."""
    m = _unsorted_tri()

    def solve_poisson(e):
        basis = Basis(m, e)
        x = solve(*condense(laplace.assemble(basis),
                            unit_load.assemble(basis),
                            D=basis.get_dofs()))
        return np.asarray(basis.interpolate(x))

    assert_allclose(solve_poisson(ElementTriP(p)), solve_poisson(elem()),
                    atol=1e-13)


def _rotated_quad():
    """A quadrilateral mesh where the element vertices are rotated."""
    m = MeshQuad().refined(2)
    rot = np.random.default_rng(1).integers(0, 4, m.t.shape[1])
    t = np.array([np.roll(m.t[:, k], r) for k, r in enumerate(rot)]).T
    return MeshQuad(m.p, t)


@pytest.mark.parametrize("p", [3, 4, 5])
def test_quad_p_facet_dof_orientation(p):
    """ElementQuadP is continuous and gives the same solution also if the
    element vertices are rotated."""
    pts = np.random.default_rng(3).random((2, 20))
    sols = []
    for m in [MeshQuad().refined(2), _rotated_quad()]:
        e = ElementQuadP(p)
        fbasis = [InteriorFacetBasis(m, e, side=i) for i in range(2)]
        x = np.random.default_rng(0).random(fbasis[0].N)
        assert_allclose(np.asarray(fbasis[0].interpolate(x)),
                        np.asarray(fbasis[1].interpolate(x)),
                        atol=1e-12)
        basis = Basis(m, e)
        x = solve(*condense(laplace.assemble(basis),
                            unit_load.assemble(basis),
                            D=basis.get_dofs()))
        sols.append(basis.probes(pts) @ x)
    assert_allclose(sols[0], sols[1], atol=1e-12)


def _facet_jumps(m, e, x):
    """Jumps across the interior facets in the conserved component."""
    fbasis = [InteriorFacetBasis(m, e, side=i) for i in range(2)]
    u = [np.asarray(fb.interpolate(x)) for fb in fbasis]
    n = fbasis[0].normals
    if isinstance(e, ElementTriHHJ1):
        return np.einsum('i...,ij...,j...', n, u[0] - u[1], n)
    if isinstance(e, (ElementTriN1, ElementTriN2, ElementTriN3)):
        return np.einsum('i...,i...', u[0] - u[1], np.array([-n[1], n[0]]))
    return np.einsum('i...,i...', u[0] - u[1], n)


@pytest.mark.parametrize(
    "m",
    [
        MeshTri().refined(2),
        _unsorted_tri(),
    ]
)
@pytest.mark.parametrize(
    "e",
    [
        ElementTriRT1(),
        ElementTriRT2(),
        ElementTriBDM1(),
        ElementTriN1(),
        ElementTriN2(),
        ElementTriN3(),
        ElementTriHHJ1(),
        ElementTriRT(3),
        ElementTriRT(5),
    ]
)
def test_tri_piola_facet_dof_orientation(m, e):
    """The normal, tangential or normal-normal component is continuous also
    if the element vertices are not sorted."""
    x = np.random.default_rng(0).random(Basis(m, e).N)
    assert_allclose(_facet_jumps(m, e, x), 0., atol=1e-10)


def test_tri_n3_facet_basis():
    """ElementTriN3 can be used with FacetBasis."""
    m = MeshTri().refined(1)
    fbasis = FacetBasis(m, ElementTriN3())
    x = np.random.default_rng(0).random(fbasis.N)
    assert np.asarray(fbasis.interpolate(x)).shape[1] == len(fbasis.find)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 6])
def test_tri_rt_facet_moments(p):
    """The facet DOFs are the moments of the normal component against the
    Legendre polynomials, defined using the global facet normal and the
    direction of increasing global vertex indices."""
    m = _unsorted_tri()
    m = MeshTri(m.p + .05 * np.sin(5. * m.p[::-1]), m.t, sort_t=False)
    e = ElementTriRT(p)
    basis = Basis(m, e)
    x = np.random.default_rng(0).random(basis.N)
    fbasis = [InteriorFacetBasis(m, e, side=i, intorder=2 * p)
              for i in range(2)]
    find = fbasis[0].find
    X = np.asarray(fbasis[0].global_coordinates())
    p0, p1 = m.p[:, m.facets[0, find]], m.p[:, m.facets[1, find]]
    s = (np.einsum('i...,i...', X - p0[:, :, None], (p1 - p0)[:, :, None])
         / np.sum((p1 - p0) ** 2, axis=0)[:, None])
    for fb in fbasis:
        un = np.einsum('i...,i...', np.asarray(fb.interpolate(x)),
                       fbasis[0].normals)
        for j in range(p):
            c = np.zeros(j + 1)
            c[j] = 1.
            moment = np.sum(un * np.polynomial.legendre.legval(2. * s - 1.,
                                                               c)
                            * fbasis[0].dx, axis=1)
            assert_allclose(moment, x[basis.facet_dofs[j, find]],
                            atol=1e-10)


@pytest.mark.parametrize("p", [1, 2, 3, 4, 5])
def test_tri_rt_divergence(p):
    """The divergence matches central finite differences."""
    e = ElementTriRT(p)
    X = np.array([[.2, .3, .1, .6],
                  [.1, .25, .7, .2]])
    eps = 1e-6
    for i in range(3 * e.facet_dofs + e.interior_dofs):
        _, d = e.lbasis(X, i)
        fd = sum((e.lbasis(X + eps * np.eye(2)[:, [c]], i)[0][c]
                  - e.lbasis(X - eps * np.eye(2)[:, [c]], i)[0][c])
                 / (2. * eps) for c in range(2))
        assert_allclose(fd, d, atol=1e-6 * max(1., np.abs(d).max()))


@pytest.mark.parametrize(
    "p,elem",
    [
        (1, ElementTriRT1),
        (2, ElementTriRT2),
    ]
)
def test_tri_rt_same_space(p, elem):
    """ElementTriRT gives the same mixed Poisson solution as the existing
    elements of the same degree."""
    m = _unsorted_tri()

    @BilinearForm
    def mixed(sig, u, tau, v, w):
        return dot(sig, tau) - div(tau) * u - div(sig) * v

    @LinearForm
    def load(tau, v, w):
        return -np.sin(np.pi * w.x[0]) * v

    sols = []
    for e in [ElementTriRT(p), elem()]:
        basis = Basis(m, e * ElementTriDG(ElementTriP(p - 1) if p > 1
                                       else ElementTriP0()))
        x = solve(mixed.assemble(basis), load.assemble(basis))
        sols.append([np.asarray(f) for f in basis.interpolate(x)])
    for f, g in zip(*sols):
        assert_allclose(f, g, atol=1e-12)
