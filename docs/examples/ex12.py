r"""Postprocessing Laplace equation.

.. note::

   This example requires the external package `pygmsh <https://pypi.org/project/pygmsh/>`_.

In this example, `pygmsh <https://pypi.org/project/pygmsh/>`_ is used to
generate a disk, replacing the default square of :class:`skfem.mesh.MeshTri`.

A basic postprocessing step in finite element analysis is evaluating linear
forms over the solution. For the Poisson equation, the integral
of the solution (normalized by the area) is the 'Boussinesq k-factor'; for
the square it's roughly 0.03514, for the circle 1/Pi/8 = 0.03979. Linear forms
are easily evaluated in skfem using the 1-D arrays assembled using the
@LinearForm decorator. In :ref:`poisson`, the linear form required for simple
integration happens to be the same one used on the right-hand side of the
differential equation, so it's already to hand.

Another is interpolation; i.e. evaluation of the solution at a
specified point which isn't necessarily a node of the mesh.  For this
problem, the maximum of the solution (normalized by the area) is the
'Boussinesq k'-factor'; by symmetry, this occurs for squares (k' =
0.07363) and circles (k' = 1/Pi/4) at the centre and so can be
evaluated by interpolation.

License
-------

Copyright 2018-2020 scikit-fem developers

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""

from skfem import *
from skfem.models.poisson import laplace, unit_load
from skfem.io import from_meshio

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

x = solve(*condense(A, b, D=basis.find_dofs()))

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
