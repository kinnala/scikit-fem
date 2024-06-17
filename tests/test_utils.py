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

