r"""Postprocessing Laplace equation.

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

"""
from pathlib import Path

from skfem import *
from skfem.models.poisson import laplace, unit_load
from skfem.io.json import from_file

import numpy as np

m = MeshTri.init_circle(4)

basis = InteriorBasis(m, ElementTriP2())

A = asm(laplace, basis)
b = asm(unit_load, basis)

x = solve(*condense(A, b, D=basis.find_dofs()))

area = sum(b)
k = b @ x / area**2
k1, = basis.probes(np.zeros((2, 1)))(x) / area

if __name__ == '__main__':
    from skfem.visuals.matplotlib import plot, show

    print('area = {:.4f} (exact = {:.4f})'.format(area, np.pi))
    print('k = {:.5f} (exact = 1/8/pi = {:.5f})'.format(k, 1/np.pi/8))
    print("k' = {:.5f} (exact = 1/4/pi = {:.5f})".format(k1, 1/np.pi/4))

    plot(basis, x)
    show()
