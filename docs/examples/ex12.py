from skfem import *
from skfem.models.poisson import laplace, unit_load
from skfem.importers import from_meshio

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

geom = Geometry()
circle = geom.add_circle([0.] * 3, 1., .5**3)
geom.add_physical(circle.line_loop.lines, 'circle')
geom.add_physical(circle.plane_surface, 'disk')
m = from_meshio(generate_mesh(geom, dim=2))

basis = InteriorBasis(m, ElementTriP2())

A = asm(laplace, basis)
b = asm(unit_load, basis)

D = basis.get_dofs(m.boundaries)
I = basis.complement_dofs(D)

x = 0*b
x = solve(*condense(A, b, I=I))

area = sum(b)
k = b @ x / area**2
k1, = basis.interpolator(x)(np.zeros((2, 1))) / area

if __name__ == '__main__':
    from skfem.visuals.matplotlib import plot3, show

    print('area = {:.4f} (exact = {:.4f})'.format(area, np.pi))
    print('k = {:.5f} (exact = 1/8/pi = {:.5f})'.format(k, 1/np.pi/8))
    print("k' = {:.5f} (exact = 1/4/pi = {:.5f})".format(k1, 1/np.pi/4))

    plot3(basis, x)
    show()
