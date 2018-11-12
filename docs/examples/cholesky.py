#!/usr/bin/env python

"""Decay of the fundamental Dirichlet mode in a rectangle.

*This example requires :module:`sksparse`.*

The solutions of the heat equation

.. math::
    \frac{\partial T}{\partial t} = \kappa\Delta T

in various tensor-product domains consist of the sum of modes which
are tensor-products of the modes on the factor-domains (Carslaw &
Jaegar 1959, §5.6).

* Carslaw, H. S., & J. C. Jaeger (1959). _Conduction of Heat in Solids_ (2nd ed.). Oxford University Press

For example, in the rectangle :math:`|x| < w_0, |y| < w_1, with
homogeneous Dirichlet boundary conditions, the modes are products of a
cosine in each direction,

.. math::
    \exp -\left\{
      \frac{\kappa\pi^2 t}{4}
      \left(
        \frac{2n_0 + 1}{w_0^2} + \frac{2n_1 + 1}{w_1^2}
      \right)
    \right\}
    \cos\frac{\pi x}{2w_0}\cos\frac{\pi y}{2w_1}   


for :math:`n_0, n_1 = 0, 1, \ldots`.

Here we simulate the decay of the fundamental, :math:`n_0 = n_1 = 0`,
discretizing time using the generalized ('theta method') trapezoidal
rule.

For a constant time-step, this leads to a linear algebraic problem at
each time with the same matrix but changing right-hand side.  This
motivates factoring the matrix; since it is symmetric in the Galerkin
finite element method, Cholesky's algorithm can be used.

An efficient sparse implementation of this is available in the
external Python package sksparse which in turn requires the CHOLMOD
library from SuiteSparse.  The latter may not be available for all
systems.

"""

from math import ceil

from matplotlib.pyplot import subplots, pause
import numpy as np
from scipy.sparse import csr_matrix

from sksparse.cholmod import cholesky

from skfem import *
from skfem.models.poisson import laplace, mass


halfwidth = np.array([2., 3.])
ncells = 2**3
diffusivity = 5.

mesh = (MeshLine(np.linspace(-1, 1, 2 * ncells) * halfwidth[0]) *
        MeshLine(np.linspace(-1, 1,
                             2 * ncells * ceil(halfwidth[1] // halfwidth[0]))
                 * halfwidth[1]))._splitquads()

element = ElementTriP1()
basis = InteriorBasis(mesh, element)

L = diffusivity * asm(laplace, basis)
M = asm(mass, basis)

dt = 1.0 * (min(halfwidth) / ncells)**2 / diffusivity
print(f'dt = {dt} µs')
theta = 0.5                     # Crank–Nicolson
A = M + theta * L * dt
B = M - (1 - theta) * L * dt

boundary = basis.get_dofs().all()
interior = basis.complement_dofs(boundary)
u = (np.cos(np.pi * mesh.p[0, :] / 2 / halfwidth[0])
     * np.cos(np.pi * mesh.p[1, :] / 2 / halfwidth[1]))

backsolve = cholesky(condense(A, D=boundary).T)  # cholesky prefers CSC


if __name__ == '__main__':
    fig, ax = subplots()
    ax.axis('off')

    t = 0.
    while True:

        u0 = {'skfem': basis.interpolator(u)(np.zeros((2, 1)))[0],
              'exact': np.exp(-diffusivity * np.pi**2 * t / 4 *
                              sum(halfwidth**-2))}
        print(','.join(map('  {:.4f}'.format,
                           [t, u0['skfem'], u0['skfem'] - u0['exact']])))
        if u0['skfem'] < 2**-4:
            break

        ax.cla()
        ax.axis('off')
        fig.suptitle('t = {:.4f}'.format(t))
        mesh.plot(u, ax=ax, zlim=(0, 1))
        if t == 0.:
            fig.colorbar(ax.get_children()[0])
        fig.show()
        pause(0.1)

        _, b1 = condense(csr_matrix(A.shape),  # ignore condensed matrix
                         B @ u, D=boundary)

        u[interior] = backsolve(b1)
        t += dt
