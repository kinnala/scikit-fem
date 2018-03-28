from skfem import *
from skfem.models import *

"""
Solve Laplace equation with zero Dirichlet BC
using trilinear hexahedral elements.
"""

m = MeshHex()
m.refine(4)

e = ElementHex1()
map = MappingIsoparametric(m, e)
basis = InteriorBasis(m, e, map, 3)

A = asm(laplace, basis)
b = asm(unit_load, basis)

I = m.interior_nodes()

x = 0*b
x[I] = solve(*condense(A, b, I=I))

m.export_vtk('hexatest', x)
