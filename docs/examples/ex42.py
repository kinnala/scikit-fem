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


m = MeshTri1DG.init_tensor(
    np.linspace(0, 1, 30),
    np.linspace(0, 1, 30),
    periodic=[0],
)

peclet = 1e2

basis = Basis(m, ElementTriP2())
A = laplace.assemble(basis) + peclet * advection.assemble(basis)
f = source.assemble(basis)

x = solve(*condense(A, f, D=basis.get_dofs()))

if __name__ == '__main__':
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import plot, savefig
    plot(basis, x, shading='gouraud', colorbar=True)
    savefig(splitext(argv[0])[0] + '_solution.png')
