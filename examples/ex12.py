"""
Author: gdmcbain

In this example 'pygmsh' is used to generate a disk, replacing the default
square of MeshTri() in ex01.py.

A basic postprocessing step in finite element analysis is evaluating linear
forms over the solution. For the boundary value problem of ex01.py, the integral
of the solution (normalized by the area) is the 'Boussinesq k-factor'; for
the square it's roughly 0.03514, for the circle 1/π/8 ≐ 0.03979. Linear forms
are easily evaluated in skfem using the 1-D arrays assembled using the
@linear_form decorator. In ex01.py, the linear form required for simple
integration happens to be the same one used on the right-hand side of the
differential equation, so it's already to hand.
"""

from skfem import *

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
geom.add_physical_surface(geom.add_circle([0.] * 3, 1., .5**3).plane_surface,
                          'disk')
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
