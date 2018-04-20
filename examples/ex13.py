# contributed by gdmcbain

from skfem import *

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
geom.add_circle([0.] * 3, 1., .5**3)
points, cells = generate_mesh(geom)[:2]
m = MeshTri(points[:, :2].T, cells['triangle'].T)

e = ElementTriP1()
map = MappingAffine(m)
basis = InteriorBasis(m, e, map, 2)

@bilinear_form
def laplace(u, du, v, dv, w):
    return du[0]*dv[0] + du[1]*dv[1]

@linear_form
def load(v, dv, w):
    return 1.0*v

A = asm(laplace, basis)
b = asm(load, basis)

I = m.interior_nodes()

x = 0*b
x[I] = solve(*condense(A, b, I=I))

area = b @ np.ones_like(x)
k = b @ x / area**2
print('area = {:.4f} (exact = {:.4f})'.format(area, np.pi))
print('k = {:.5f} (exact = 1/8/pi = {:.5f})'.format(k, 1/np.pi/8))

m.plot3(x)
m.show()
