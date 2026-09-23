import numpy as np
from numpy.testing import assert_allclose

from skfem import Basis, ElementTriMorley, FacetBasis, MeshTri2


def test_curved_boundary_normal_projection():
    mesh = MeshTri2.init_circle(1)
    element = ElementTriMorley()
    basis = Basis(mesh, element)
    boundary = FacetBasis(mesh, mesh.elem(), intorder=0)
    vertices = mesh.mapping().F(element.refdom.p)
    # A quadratic polynomial has exactly reproducible Morley DOFs.
    coefficients = np.empty((6, mesh.nelements))
    for local in range(3):
        x, y = vertices[:, :, local]
        coefficients[local] = x*x + 2.*x*y + 3.*y*y
        edge = element.refdom.facets[local]
        tangent = vertices[:, :, edge[0]] - vertices[:, :, edge[1]]
        normal = np.array([tangent[1], -tangent[0]])
        normal /= np.linalg.norm(normal, axis=0)
        for position, facet in enumerate(boundary.find):
            cell = boundary.tind[position]
            if mesh.t2f[local, cell] == facet:
                normal[:, cell] *= (normal[:, cell]
                                    @ boundary.normals[:, position, 0])
        x, y = vertices[:, :, edge].mean(axis=2)
        coefficients[local + 3] = (normal[0] * (2.*x + 2.*y)
                                   + normal[1] * (2.*x + 6.*y))
    value = sum(c[:, None] * shape
                for c, (shape,) in zip(coefficients, basis.basis))
    x, y = basis.global_coordinates()
    assert_allclose(value, x*x + 2.*x*y + 3.*y*y, rtol=1e-12, atol=1e-12)
