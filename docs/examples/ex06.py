"""
Author: kinnala

Visualising high-order solutions by refining the mesh and
interpolating the solution.
"""
from skfem import *
from skfem.models.poisson import laplace, unit_load
import numpy as np

m = MeshQuad()
m.refine(2)

e1 = ElementQuad1()
e = ElementQuad2()
map = MappingIsoparametric(m, e1)
ib = InteriorBasis(m, e, map, 4)

K = asm(laplace, ib)

f = asm(unit_load, ib)

D = ib.get_dofs().all()
x = np.zeros(ib.N)
I = ib.complement_dofs(D)

x[I] = solve(*condense(K, f, D=D))

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    ax = m.draw()
    M.plot(X, smooth=True, edgecolors='', ax=ax)
    M.show()
