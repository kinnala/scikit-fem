"""Periodic meshes.

When working with periodic boundary conditions, it may be more convenient to
use a periodic mesh rather than to explicitly force the periodicity on the
linear algebra level.  This example defines a "discontinuous mesh topology"
using :class:`~skfem.mesh.MeshTri1DG`.  In practice this means that the
local-to-global reference mapping is not done through an affine mapping but
through the isoparametric mapping of a discontinuous P1 finite element.
In the mesh data structure the vertices of the mesh are repeated even if shared
across multiple elements through the degrees-of-freedom.

In this example we solve the advection equation with a Gaussian source in the
middle of a unit square.  The mesh is periodic from right-to-left so that the
resulting solution is also periodic.

"""
import numpy as np

from skfem import *
from skfem.models import laplace


@BilinearForm
def advection(u, v, _):
    return v * u.grad[0]


@LinearForm
def source(v, w):
    x = w.x - .5
    return v * np.exp(-1e3 * (x[0] ** 2 + x[1] ** 2))


# nonperiodic mesh
m = MeshTri.init_symmetric().refined(5)

# create a periodic mesh
Mp = MeshTri1DG.periodic(
    m,
    m.nodes_satisfying(lambda x: x[0] == 1),
    m.nodes_satisfying(lambda x: x[0] == 0),
)

peclet = 1e2

basis = Basis(Mp, ElementTriP2())
A = laplace.assemble(basis) + peclet * advection.assemble(basis)
f = source.assemble(basis)

D = basis.get_dofs()
x = solve(*condense(A, f, D=D))

if __name__ == '__main__':
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import plot, savefig
    plot(basis, x, shading='gouraud')
    savefig(splitext(argv[0])[0] + '_solution.png')
