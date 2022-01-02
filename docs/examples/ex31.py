r"""# Curved elements

This example solves the eigenvalue problem

.. math::
   -\Delta u = \lambda u \quad \text{in $\Omega$},
with the boundary condition :math:`u|_{\partial \Omega} = 0` using isoparametric
mapping via biquadratic basis and finite element approximation using fifth-order
quadrilaterals.

"""
from skfem import *
from skfem.models.poisson import laplace, mass
import numpy as np


p = np.array([[0.  ,  1.  ,  1.  ,  0.  ,  0.5 ,  0.  ,  1.  ,  0.5 ,  0.5 ,
               0.25, -0.1 ,  0.75,  0.9 ,  1.1 ,  0.75,  0.1 ,  0.25,  0.5 ,
               0.25,  0.75,  0.5 ,  0.25,  0.75,  0.75,  0.25],
              [0.  ,  0.  ,  1.  ,  1.  ,  0.  ,  0.5 ,  0.5 ,  1.  ,  0.5 ,
               0.1 ,  0.25, -0.1 ,  0.25,  0.75,  0.9 ,  0.75,  1.1 ,  0.25,
               0.5 ,  0.5 ,  0.75,  0.25,  0.25,  0.75,  0.75]])

t = np.array([[ 0,  4,  8,  5],
              [ 4,  1,  6,  8],
              [ 8,  6,  2,  7],
              [ 5,  8,  7,  3],
              [ 9, 11, 19, 18],
              [17, 12, 13, 20],
              [18, 19, 14, 16],
              [10, 17, 20, 15],
              [21, 22, 23, 24]])

m = MeshQuad2(p, t)
e = ElementQuadP(5)

# create mapping for the finite element approximation and assemble
basis = Basis(m, e)

A = asm(laplace, basis)
M = asm(mass, basis)

L, x = solve(*condense(A, M, D=basis.get_dofs()), solver=solver_eigen_scipy_sym(k=8))

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv
    name = splitext(argv[0])[0]

    from skfem.visuals.matplotlib import *
    ax = draw(m)
    plot(basis, x[:, 6], Nrefs=6, ax=ax)
    savefig(f'{name}_eigenmode.png')
