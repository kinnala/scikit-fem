from skfem import *
from skfem.helpers import dot, grad
from scipy.sparse.linalg import spsolve

# # enable additional mesh validity checks, sacrificing performance
# import logging
# logging.basicConfig(format='%(levelname)s %(asctime)s %(name)s %(message)s')
# logging.getLogger('skfem').setLevel(logging.DEBUG)

# create the mesh
m = MeshTri().refined(6)
# or, with your own points and cells:
# m = MeshTri(points, cells)

e = ElementTriP1()
basis = Basis(m, e)

# this method could also be imported from skfem.models.laplace
@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


# this method could also be imported from skfem.models.unit_load
@LinearForm
def rhs(v, _):
    return 1.0 * v

A = asm(laplace, basis)
b = asm(rhs, basis)
# or:
# A = laplace.assemble(basis)
# b = rhs.assemble(basis)

# enforce Dirichlet boundary conditions and solve
x = bc(spsolve, D=m.boundary_nodes())(A, b)


def visualize():
    from skfem.visuals.matplotlib import plot
    return plot(m, x, shading='gouraud', colorbar=True)

if __name__ == "__main__":
    visualize().show()
