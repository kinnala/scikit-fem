import pytest

from numpy.testing import assert_array_almost_equal
from skfem.assembly import Basis, BilinearForm
from skfem.element import ElementTriP1, ElementVector
from skfem.mesh import MeshTri, MeshTet
from skfem.experimental.lab import symbols


@pytest.mark.parametrize(
    "form,m,elem",
    [
        ("dot(grad(u), grad(v))", MeshTri().refined(), ElementTriP1()),
        ("u * v", MeshTri().refined(), ElementTriP1()),
        ("u[0] * v[0]", MeshTri().refined(), ElementVector(ElementTriP1())),
        ("ddot(transpose(grad(u)), grad(v))", MeshTri().refined(), ElementVector(ElementTriP1())),
    ]
)
def test_compare_linear_forms(form, m, elem):

    ibasis = Basis(m, elem)
    fbasis = ibasis.boundary()

    @BilinearForm
    def form1(u, v, w):
        from skfem.helpers import dot, grad, ddot, transpose
        return eval(form)

    def form2():
        from skfem.experimental.lab import dot, grad, ddot, transpose
        u, v, x, h, n = symbols(elem, x=True, h=True, n=True)
        return eval(form)

    for basis in [ibasis, fbasis]:
        A1 = BilinearForm(form1).assemble(basis).todense()
        A2 = form2().assemble(basis).todense()
        A3, _ = form2().assemble(basis, x=basis.zeros())
        A3 = A3.todense()

        assert_array_almost_equal(A1, A2)
        assert_array_almost_equal(A1, A3)
