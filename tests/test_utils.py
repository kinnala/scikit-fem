from unittest import TestCase

import numpy as np
import scipy.sparse
from numpy.testing import assert_almost_equal

from skfem.assembly import CellBasis, Basis, LinearForm, asm, BilinearForm
from skfem.element import ElementTriP1, ElementQuad1, ElementTriP2
from skfem.mesh import MeshTri, MeshQuad
from skfem.utils import projection, enforce, condense, solve, mpc
from skfem.models import laplace, mass, unit_load


class InitializeScalarField(TestCase):

    def runTest(self):

        mesh = MeshTri().refined(5)
        basis = CellBasis(mesh, ElementTriP1())

        def fun(X):
            x, y = X
            return x ** 2 + y ** 2

        x = projection(fun, basis)
        y = fun(mesh.p)

        normest = np.linalg.norm(x - y)

        self.assertTrue(normest < 0.011,
                        msg="|x-y| = {}".format(normest))


class TestEnforce(TestCase):

    mesh = MeshTri()

    def runTest(self):

        m = self.mesh
        e = ElementTriP1()
        basis = CellBasis(m, e)

        A = laplace.assemble(basis)
        M = mass.assemble(basis)
        D = m.boundary_nodes()

        assert_almost_equal(enforce(A, D=D).toarray(), np.eye(A.shape[0]))
        assert_almost_equal(enforce(M, D=D, diag=0.).toarray(),
                            np.zeros(M.shape))

        enforce(A, D=D, overwrite=True)
        assert_almost_equal(A.toarray(), np.eye(A.shape[0]))


def test_enforce_handles_empty_rows():
    """Regression: ``enforce`` must work when some enforced rows have no
    nonzeros to begin with -- this happens whenever an assembled matrix has
    rows belonging to elements that contribute only Dirichlet dofs (e.g.
    a p-Laplacian Hessian whose surface dofs see no neighbours through the
    bulk form). The previous implementation built an offset array via
    ``np.cumsum(np.ones(...))`` and silently produced out-of-range indices
    on any zero-row, raising ``IndexError``. See issue #1195.
    """
    m = MeshTri().refined(1)
    basis = CellBasis(m, ElementTriP1())

    # Build a matrix with at least one entirely-zero row among the boundary
    # dofs by assembling a bilinear form whose coefficient vanishes
    # everywhere -- this mirrors what happens when surface dofs end up
    # with no in-bulk neighbours.
    @BilinearForm
    def zero_form(u, v, w):
        return 0. * u * v

    A = zero_form.assemble(basis)
    D = m.boundary_nodes()

    # Confirm we really have empty rows in D (otherwise the test would
    # silently pass even with the old buggy code).
    counts = A.indptr[D + 1] - A.indptr[D]
    assert (counts == 0).any(), "test setup did not produce any empty rows"

    Aout = enforce(A, D=D)

    # Every enforced row must end up with diag = 1 and no off-diagonal
    # nonzeros; non-enforced rows must be untouched.
    dense = Aout.toarray()
    for d in D:
        row = dense[d]
        assert row[d] == 1.0
        row[d] = 0.0
        assert not row.any()
    rest = np.setdiff1d(np.arange(A.shape[0]), D)
    assert_almost_equal(dense[rest], A.toarray()[rest])

    # And with a right-hand side ``b`` the enforced entries must be set.
    b = np.ones(A.shape[0])
    _, bout = enforce(A, b, D=D)
    assert_almost_equal(bout[D], 0.)
    assert_almost_equal(bout[rest], b[rest])


def test_simple_cg_solver():

    m = MeshTri().refined(3)
    basis = CellBasis(m, ElementTriP1())

    A0 = laplace.coo_data(basis)
    A1 = laplace.assemble(basis)

    f = unit_load.assemble(basis)

    D = m.boundary_nodes()

    x1 = solve(*condense(A1, f, D=D))

    f[D] = 0

    x0 = A0.solve(f, D=D)

    assert_almost_equal(x0, x1)


def test_mpc_periodic():

    m = MeshQuad().refined(3).with_defaults()
    basis = Basis(m, ElementQuad1())
    A = laplace.assemble(basis)
    b = unit_load.assemble(basis)
    y = solve(*mpc(A, b,
                   S=np.concatenate((
                       m.nodes_satisfying(lambda x: (x[0] == 0) * (x[1] < 1) * (x[1] > 0)),
                       basis.get_dofs({'top', 'bottom'}),
                   )),
                   M=m.nodes_satisfying(lambda x: (x[0] == 1) * (x[1] < 1) * (x[1] > 0))))

    assert_almost_equal(y[basis.get_dofs('left')], y[basis.get_dofs('right')])
    assert_almost_equal(y[basis.get_dofs('left')], y[basis.get_dofs(lambda x: x[0] == .5)])


def test_mpc_2x_periodic():

    m = MeshTri.init_sqsymmetric().refined(4).with_defaults()
    e = ElementTriP2()

    basis = Basis(m, e)

    left = basis.get_dofs(facets='left').sort()[1:-1]
    right = basis.get_dofs(facets='right').sort()[1:-1]
    dirichlet = basis.get_dofs(facets={'top', 'bottom'})

    A = asm(laplace, basis)
    f = asm(LinearForm(lambda v, w: 1. * v), basis)

    # doubly periodic
    M = left
    S = np.concatenate((right, dirichlet))
    T = 2. * scipy.sparse.eye(len(S), len(M)).tocsr()

    y = solve(*mpc(A, f, M=M, S=S, T=T))

    assert_almost_equal(2 * y[basis.get_dofs('left')], y[basis.get_dofs('right')])



def test_mpc_doubly_periodic():

    m = MeshTri.init_sqsymmetric().refined(5).with_defaults()
    e = ElementTriP1()

    basis = Basis(m, e)

    left = basis.get_dofs(facets='left').sort()
    right = basis.get_dofs(facets='right').sort()
    top = basis.get_dofs(facets='top').sort()
    bottom = basis.get_dofs(facets='bottom').sort()

    topleft = basis.get_dofs(nodes=(0, 1))
    others = basis.get_dofs(nodes=[(1, 1), (1, 0), (0, 0)])
    A = asm(laplace, basis)
    f = asm(LinearForm(lambda v, w: (np.sin(2. * np.pi * w.x[0])
                                     + np.sin(2. * np.pi * w.x[1])) * v), basis)

    # doubly periodic
    M = np.concatenate((
        left[1:-1],
        top[1:-1],
        topleft,
    ))
    S = np.concatenate((
        right[1:-1],
        bottom[1:-1],
        others
    ))
    T = scipy.sparse.eye(len(S), len(M)).tocsr()
    T[-1, -1] = 1
    T[-2, -1] = 1
    T[-3, -1] = 1
    # periodic x2
    # M = right
    # S = np.concatenate((left, top, bottom))
    # T = 2 * scipy.sparse.eye(len(S), len(M))

    y = solve(*mpc(A, f, M=M, S=S, T=T))

    assert_almost_equal(y[basis.get_dofs('left')], y[basis.get_dofs('right')])
    assert_almost_equal(y[basis.get_dofs('bottom')], y[basis.get_dofs('top')])


def test_mpc_dirichlet():

    m = MeshTri().refined(3)
    basis = Basis(m, ElementTriP1())
    A = laplace.assemble(basis)
    b = unit_load.assemble(basis)
    D = basis.get_dofs()
    # these two should be equal
    y1 = solve(*mpc(A, b, S=D))
    y2 = solve(*condense(A, b, D=D))
    assert_almost_equal(y1, y2)
    # inhomogeneous
    y1 = solve(*mpc(A, b, S=D, g=m.p[0, D]))
    y2 = solve(*condense(A, b, D=D, x=m.p[0]))
    assert_almost_equal(y1, y2)

