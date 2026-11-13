from skfem import *
from skfem.helpers import dot, grad


# create the mesh
m = MeshTri().refined(6)
# or, with your own points and cells:
# m = MeshTri(points, cells)
# or, load from file
# m = MeshTri.load("mesh.msh")

e = ElementTriP1()
basis = Basis(m, e)


@BilinearForm
def laplace(u, v, _):
    return dot(grad(u), grad(v))


@LinearForm
def rhs(v, _):
    return 1.0 * v


A = laplace.assemble(basis)
b = rhs.assemble(basis)

# enforce Dirichlet boundary conditions
A, b = enforce(A, b, D=m.boundary_nodes())

# solve -- can be anything that takes a sparse matrix and a right-hand side
x = solve(A, b)

def visualize():
    from skfem.visuals.web import plot
    return plot(basis, x, warp=True, edges=False, cmap='jet')

if __name__ == "__main__":
    visualize().show()
