"""Tests run in conda/mamba env due to challenging binary deps."""

import pytest
from unittest import TestCase, main
from numpy.testing import assert_almost_equal


@pytest.mark.mpi_skip
class TestEx52(TestCase):

    def runTest(self):
        import docs.examples.ex52 as ex52
        # the solver tolerance is rtol=1e-6
        self.assertAlmostEqual(ex52.xmax, 0.999610970807614, places=5)


@pytest.mark.mpi
def test_ex_53():

    import docs.examples.ex53 as ex53
    # the solver tolerance is rtol=1e-6
    assert_almost_equal(ex53.xmax, 0.9996111271291415, decimal=5)


@pytest.mark.mpi
def test_ex_54():

    import docs.examples.ex54 as ex54
    assert_almost_equal(ex54.xmax, 1.692582561905199, decimal=5)


def _poisson_serial_vs_distributed(mesh, elem, facet=False, use_condense=False):
    import numpy as np
    from skfem import (Basis, FacetBasis, BilinearForm, LinearForm,
                       Functional, solve, enforce, condense, solver_petsc)
    from skfem.helpers import inner, grad

    @BilinearForm
    def a(u, v, w):
        return inner(grad(u), grad(v)) + inner(u, v)

    @BilinearForm
    def robin(u, v, w):
        return 2. * inner(u, v)

    @LinearForm
    def f(v, w):
        load = np.sin(w.x[0]) + w.x[1]
        return load * (v if len(v.shape) == 2 else v[0])

    @Functional
    def l2(w):
        return inner(w['u'], w['u'])

    def compute(m):
        basis = Basis(m, elem)
        A = a.assemble(basis)
        b = f.assemble(basis)
        if facet:
            A = A + robin.assemble(FacetBasis(m, elem))
            D = basis.get_dofs('left')
        else:
            D = basis.get_dofs()
        # nonzero boundary values
        g = basis.zeros()
        g[D] = 1. + basis.doflocs[0, D] ** 2
        if use_condense:
            x = solve(*condense(A, b, x=g, D=D),
                      solver=solver_petsc(ksp_type='gmres', rtol=1e-12))
        else:
            x = solve(*enforce(A, b, x=g, D=D),
                      solver=solver_petsc(ksp_type='gmres', rtol=1e-12))
        return basis, x, l2.assemble(basis, u=basis.interpolate(x))

    _, xs, ns = compute(mesh)
    basis, xd, nd = compute(mesh.distributed())
    np.testing.assert_allclose(xd, xs[basis.dofs._globnums], atol=1e-8)
    np.testing.assert_allclose(nd, ns, rtol=1e-8)


@pytest.mark.mpi
@pytest.mark.parametrize(
    "mtype,etype,kwargs",
    [
        ('MeshTri', 'ElementTriP1', {}),
        ('MeshTri', 'ElementTriP2', {'use_condense': True}),
        ('MeshTri', 'ElementTriMini', {'facet': True}),
        ('MeshQuad', 'ElementQuad2', {'facet': True}),
        ('MeshTet', 'ElementTetP2', {}),
        ('MeshTet', 'ElementTetP2', {'facet': True, 'use_condense': True}),
        ('MeshHex', 'ElementHex2', {}),
        ('MeshTri', ('ElementVector', 'ElementTriP2'), {}),
    ]
)
def test_distributed_matches_serial(mtype, etype, kwargs):
    import skfem
    mesh = getattr(skfem, mtype)().refined(3 if mtype.endswith(('Tri', 'Quad'))
                                          else 2).with_defaults()
    if isinstance(etype, tuple):
        elem = getattr(skfem, etype[0])(getattr(skfem, etype[1])())
    else:
        elem = getattr(skfem, etype)()
    _poisson_serial_vs_distributed(mesh, elem, **kwargs)


@pytest.mark.mpi
def test_distributed_boundary():
    import numpy as np
    from skfem import MeshTet, distribute, allreduce
    m = MeshTet().refined(3)
    md = distribute(m)
    # only the true boundary, no interfaces between processes
    p = md.p[:, md.facets[:, md.boundary_facets()]]
    assert np.all(np.min(np.abs(np.stack((p, 1 - p))), axis=0).min(axis=0)
                  < 1e-12)
    assert (allreduce(len(md.boundary_facets()))
            == len(m.boundary_facets()))


@pytest.mark.mpi
def test_deprecated_decompose():
    import numpy as np
    import petsc4py.PETSc as petsc
    from skfem import MeshTri, Basis, Dofs, ElementTriP1, allreduce
    from skfem.models import laplace, unit_load

    comm = petsc.COMM_WORLD

    with pytest.warns(DeprecationWarning):
        @Dofs.decompose(comm)
        def builder():
            m = MeshTri().refined(3).with_defaults()
            return m, Dofs(m, ElementTriP1())

    m, dofs = builder()
    basis = Basis(m, ElementTriP1())
    with pytest.warns(DeprecationWarning):
        A = laplace.elemental(basis).topetsc(dofs)
    with pytest.warns(DeprecationWarning):
        b = unit_load.elemental(basis).topetsc(dofs)
    with pytest.warns(DeprecationWarning):
        D = dofs.l2g(basis.get_dofs({'left', 'right', 'top', 'bottom'}))
    x = A.createVecRight()
    A.zeroRowsColumns(D, diag=1., x=x, b=b)
    ksp = petsc.KSP().create()
    ksp.setOperators(A)
    ksp.setTolerances(rtol=1e-12)
    ksp.solve(b, x)
    with pytest.warns(DeprecationWarning):
        xloc = dofs.loc(x)
    assert xloc.shape == (basis.N,)
    assert np.isclose(allreduce(xloc.max(), op='max'), x.max()[1])


@pytest.mark.mpi
@pytest.mark.parametrize(
    "mtype,elem",
    [
        ('MeshTri', lambda s: s.ElementTriRT0()),
        ('MeshTri', lambda s: s.ElementTriRT(1)),
        ('MeshQuad', lambda s: s.ElementQuadRT0()),
        ('MeshTet', lambda s: s.ElementTetRT0()),
        ('MeshTet', lambda s: s.ElementTetN0()),
        ('MeshTri', lambda s: s.ElementTriMorley()),
        ('MeshTri', lambda s: s.ElementTriArgyris()),
        ('MeshTri', lambda s: s.ElementTriP(4)),
        ('MeshTri', lambda s: s.ElementDG(s.ElementTriP1())),
        ('MeshTri', lambda s: (s.ElementVector(s.ElementTriP2())
                               * s.ElementTriP1())),
    ]
)
def test_distributed_matrix_matches_serial(mtype, elem):
    import numpy as np
    import skfem
    from skfem import Basis, BilinearForm, asm, allreduce
    from skfem.helpers import inner
    from skfem.distributed import to_petsc_vec, to_local

    @BilinearForm
    def mass(*args):
        # works also for composite elements
        n = len(args[:-1]) // 2
        return sum(inner(u, v) for u, v in zip(args[:n], args[n:-1]))

    m = getattr(skfem, mtype)().refined(3 if mtype != 'MeshTet' else 2)
    e = elem(skfem)
    As = mass.assemble(Basis(m, e))
    basis = Basis(m.distributed(), e)
    glob = basis.dofs._globnums
    x = np.random.RandomState(0).rand(As.shape[0])
    for A in (mass.assemble(basis), asm(mass, basis)):
        y = A.createVecLeft()
        A.mult(to_petsc_vec(A, x[glob]), y)
        err = np.abs(to_local(A, y) - (As @ x)[glob]).max()
        assert allreduce(err, op='max') < 1e-10


@pytest.mark.mpi
def test_distributed_unsupported():
    from skfem import (MeshTri, MeshTri2, InteriorFacetBasis, Basis,
                       ElementTriP1, penalize, condense)
    from skfem.models import laplace, unit_load
    import petsc4py.PETSc as petsc

    with pytest.raises(NotImplementedError):
        MeshTri2.init_circle(1).distributed()

    m = MeshTri().refined(3).distributed()
    if petsc.COMM_WORLD.size > 1:
        with pytest.raises(NotImplementedError):
            InteriorFacetBasis(m, ElementTriP1())
    basis = Basis(m, ElementTriP1())
    A = laplace.assemble(basis)
    b = unit_load.assemble(basis)
    with pytest.raises(NotImplementedError):
        penalize(A, b, D=basis.get_dofs())
    with pytest.raises(NotImplementedError):
        condense(A, D=basis.get_dofs())  # without the right hand side
