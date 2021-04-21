r"""Point source.

Sources concentrated at points cannot be evaluated in the usual way, which 
involves discrete quadrature; instead, it requires direct use of the basis
functions, as implemented in `InteriorBasis.point_source`.

Here this is demonstrated for a disk with homogeneous Dirichlet conditions.
The exact solution is the well-known Green's function (e.g. Sadybekov,
Turmetov, & Torebek 2015).

* Sadybekov, M. A., Turmetov, B. K. & Torebek, B. T. (2015). On an explicit
  form of the Green function of the Robin problem for the Laplace operator
  in a circle. *Advances in Pure and Applied Mathematics,* 6, 163-172.
  [doi: 10.1515/apam-2015-0003](https://doi.org/10.1515%2fapam-2015-0003)



"""
from pathlib import Path

from skfem import *
from skfem.models.poisson import laplace, unit_load
from skfem.io.json import from_file

import numpy as np


m = MeshTri.init_circle(4)

basis = InteriorBasis(m, ElementTriP2())

A = asm(laplace, basis)
b = basis.point_source(np.array([0.3, 0.2]))

x = solve(*condense(A, b, D=basis.find_dofs()))

if __name__ == '__main__':
    from skfem.visuals.matplotlib import plot, show

    plot(basis, x)
    show()
