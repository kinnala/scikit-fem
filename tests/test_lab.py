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
    ]
)
def test_compare_forms(form, m, elem):

    basis = Basis(m, elem)

    @BilinearForm
    def form1(u, v, w):
        from skfem.helpers import dot, grad
        return eval(form)

    def form2():
        from skfem.experimental.lab import dot, grad
        u, v = symbols(elem)
        return eval(form)

    A1 = BilinearForm(form1).assemble(basis).todense()
    A2 = form2().assemble(basis).todense()

    assert_array_almost_equal(A1, A2)
