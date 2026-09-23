import numpy as np
import pytest
from numpy.testing import assert_allclose

from skfem import (Basis, BilinearForm, ElementQuadBFS, ElementTriArgyris,
                   MeshQuad, MeshQuad1DG, MeshTri, MeshTri1DG)
from skfem.helpers import dd, ddot
from skfem.models.poisson import laplace, mass


@BilinearForm
def hessian(u, v, w):
    return ddot(dd(u), dd(v))


def periodic_mesh(element, periodic, irregular, renumber):
    x = (np.array([0., .15, .4, .7, 1.])
         if irregular else np.linspace(0., 1., 5))
    y = np.array([0., .2, .5, .8, 1.]) if irregular else x
    if irregular:
        x, y = x + .25, y - .25
    triangle = element is ElementTriArgyris
    mesh_type = MeshTri if triangle else MeshQuad
    dg_type = MeshTri1DG if triangle else MeshQuad1DG
    mesh = mesh_type.init_tensor(x, y)
    if renumber:
        order = np.arange(mesh.nvertices)
        order[:len(y)] = order[:len(y)][::-1]
        mesh = mesh_type(mesh.p[:, order], np.argsort(order)[mesh.t])
    eliminate, keep = [], []
    for dim, values in enumerate((x, y)):
        if dim not in periodic:
            continue
        left = np.flatnonzero(mesh.p[dim] == values.min())
        right = np.flatnonzero(mesh.p[dim] == values.max())
        eliminate.extend(left[np.argsort(mesh.p[1 - dim, left])])
        keep.extend(right[np.argsort(mesh.p[1 - dim, right])])
    eliminate, unique = np.unique(eliminate, return_index=True)
    keep = np.asarray(keep)[unique]
    for index, vertex in enumerate(eliminate):
        keep[keep == vertex] = keep[index]
    return dg_type.periodic(mesh, eliminate, keep)


def facet_normal(mesh, facet):
    # The first adjacent triangle defines an interior facet's direction.
    # Boundary DOFs use the outward direction instead.
    cell = mesh.f2t[0, facet]
    local = np.flatnonzero(mesh.t2f[:, cell] == facet)[0]
    vertices = mesh.mapping().F(
        mesh.elem.refdom.p, tind=np.array([cell]))[:, 0]
    endpoints = mesh.elem.refdom.facets[local]
    tangent = vertices[:, endpoints[0]] - vertices[:, endpoints[1]]
    normal = np.array([tangent[1], -tangent[0]])
    normal /= np.linalg.norm(normal)
    if mesh.f2t[1, facet] == -1:
        away = vertices[:, endpoints].mean(axis=1) - vertices.mean(axis=1)
        normal *= np.sign(normal @ away)
    return normal


def restriction(ordinary, periodic):
    mesh = periodic.mesh
    result = np.zeros((ordinary.N, periodic.N))
    for vertex, identified in enumerate(mesh._ix):
        result[ordinary.nodal_dofs[:, vertex],
               periodic.nodal_dofs[:, identified]] = 1.
    if periodic.elem.facet_dofs:
        lookup = {tuple(sorted(f)): j for j, f in enumerate(mesh.facets.T)}
        for facet, vertices in enumerate(ordinary.mesh.facets.T):
            mapped = lookup[tuple(sorted(mesh._ix[vertices]))]
            sign = np.sign(facet_normal(ordinary.mesh, facet)
                           @ facet_normal(mesh, mapped))
            result[ordinary.facet_dofs[:, facet],
                   periodic.facet_dofs[:, mapped]] = sign
    return result


def traces(basis, coefficients):
    mesh = basis.mesh
    facets = np.flatnonzero(mesh.f2t[1] != -1)
    samples = np.linspace(0., 1., 5)
    result = []
    for side in (0, 1):
        cells = mesh.f2t[side, facets]
        local = np.array([np.argmax(mesh.t[:, cells] == nodes, axis=0)
                          for nodes in mesh.facets[:, facets]])
        X = (basis.elem.refdom.p[:, local[0], None] * (1. - samples)
             + basis.elem.refdom.p[:, local[1], None] * samples)
        edge_basis = Basis(mesh, basis.elem, elements=cells,
                           quadrature=(X, np.ones(len(samples))))
        field = edge_basis.interpolate(coefficients)
        result.append(np.array([field, *field.grad]))
    return result


@pytest.mark.parametrize('element', [ElementTriArgyris, ElementQuadBFS])
@pytest.mark.parametrize('periodic', [[0], [1], [0, 1]])
@pytest.mark.parametrize('irregular', [False, True])
@pytest.mark.parametrize('renumber', [False, True])
def test_periodic_c1(element, periodic, irregular, renumber):
    mesh = periodic_mesh(element, periodic, irregular, renumber)
    ordinary = Basis(mesh._orig, element(), intorder=10)
    basis = Basis(mesh, element(), intorder=10)
    R = restriction(ordinary, basis)
    for form in (mass, laplace, hessian):
        expected = R.T @ form.assemble(ordinary) @ R
        assert_allclose(form.assemble(basis).toarray(), expected,
                        rtol=1e-7, atol=1e-8)
    coefficients = np.random.default_rng(902).normal(size=basis.N)
    left, right = traces(basis, coefficients)
    assert_allclose(left, right, rtol=1e-7, atol=1e-8)
    left, right = traces(ordinary, R @ coefficients)
    assert_allclose(left, right, rtol=1e-7, atol=1e-8)

    nodes = Basis(mesh, basis.elem,
                  quadrature=(basis.elem.refdom.p,
                              np.ones(basis.elem.refdom.p.shape[1])))
    field = nodes.interpolate(coefficients)
    values = [field, *field.grad]
    if element is ElementTriArgyris:
        values.extend([field.hess[0, 0], field.hess[0, 1], field.hess[1, 1]])
    else:
        values.append(field.hess[0, 1])
    for row, actual in enumerate(values):
        expected = coefficients[basis.nodal_dofs[row, mesh.t]].T
        assert_allclose(actual, expected, rtol=1e-7, atol=1e-8)


@pytest.mark.parametrize('element', [ElementTriArgyris, ElementQuadBFS])
@pytest.mark.parametrize('periodic', [[0], [1], [0, 1]])
def test_periodic_local_polynomial(element, periodic):
    from numpy.polynomial.polynomial import polyder, polyval2d

    mesh = periodic_mesh(element, periodic, True, True)
    basis = Basis(mesh, element(), intorder=10)
    polynomial = np.zeros((6, 6))
    polynomial[0, 0] = polynomial[1, 1] = 1.
    derivatives = [(0, 0), (1, 0), (0, 1)]
    if element is ElementTriArgyris:
        polynomial[5, 0] = polynomial[0, 5] = polynomial[3, 2] = 1.
        derivatives.extend([(2, 0), (1, 1), (0, 2)])
    else:
        polynomial[3, 3] = 1.
        derivatives.append((1, 1))

    def evaluate(x, y, dx=0, dy=0):
        derived = polyder(polyder(polynomial, dx, axis=0), dy, axis=1)
        return polyval2d(x, y, derived)

    vertices = mesh.mapping().F(basis.elem.refdom.p)
    coefficients = []
    for local in range(vertices.shape[2]):
        x, y = vertices[:, :, local]
        coefficients.extend(evaluate(x, y, *d) for d in derivatives)
    if basis.elem.facet_dofs:
        normals = np.array([facet_normal(mesh, f)
                            for f in range(mesh.nfacets)]).T
        for local, edge in enumerate(basis.elem.refdom.facets):
            x, y = vertices[:, :, edge].mean(axis=2)
            normal = normals[:, mesh.t2f[local]]
            coefficients.append(normal[0] * evaluate(x, y, 1, 0)
                                + normal[1] * evaluate(x, y, 0, 1))
    # Local coefficients do not impose a nonperiodic polynomial on shared DOFs.
    value = sum(c[:, None] * f for c, (f,) in zip(coefficients, basis.basis))
    gradient = sum(c[None, :, None] * f.grad
                   for c, (f,) in zip(coefficients, basis.basis))
    x, y = basis.global_coordinates()
    assert_allclose(value, evaluate(x, y), rtol=1e-7, atol=1e-8)
    assert_allclose(gradient, [evaluate(x, y, 1, 0), evaluate(x, y, 0, 1)],
                    rtol=1e-7, atol=1e-8)
