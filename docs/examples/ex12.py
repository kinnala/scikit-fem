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

Another is interpolation; i.e. evaluation of the solution at a
specified point which isn't necessarily a node of the mesh.  For this
problem, the maximum of the solution (normalized by the area) is the
'Boussinesq k'-factor'; by symmetry, this occurs for squares (k' ≐
0.07363) and circles (k' = 1/π/4) at the centre and so can be
evaluated by interpolation.

"""

from skfem import *
from skfem.models.poisson import laplace, unit_load

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
geom.add_physical_surface(geom.add_circle([0.] * 3, 1., .5**3).plane_surface,
                          'disk')
points, cells = generate_mesh(geom)[:2]
m = MeshTri(points[:, :2].T, cells['triangle'].T)

basis = InteriorBasis(m, ElementTriP2())

A = asm(laplace, basis)
b = asm(unit_load, basis)

D = basis.get_dofs().all()
I = basis.complement_dofs(D)

x = 0*b
x[I] = solve(*condense(A, b, I=I))

area = sum(b)
k = b @ x / area**2
k1, = basis.interpolator(x)(np.zeros((2, 1))) / area

if __name__ == '__main__':
    print('area = {:.4f} (exact = {:.4f})'.format(area, np.pi))
    print('k = {:.5f} (exact = 1/8/pi = {:.5f})'.format(k, 1/np.pi/8))
    print("k' = {:.5f} (exact = 1/4/pi = {:.5f})".format(k1, 1/np.pi/4))

    m.plot3(x[basis.nodal_dofs.flatten()])
    m.show()
