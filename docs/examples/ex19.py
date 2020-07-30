r"""Heat equation.

The solutions of the heat equation

.. math::
    \frac{\partial T}{\partial t} = \kappa\Delta T
in various tensor-product domains consist of the sum of modes which
are tensor-products of the modes on the factor-domains (Carslaw &
Jaegar 1959, §5.6).

* Carslaw, H. S., & J. C. Jaeger (1959). Conduction of Heat in Solids (2nd ed.). Oxford University Press

For example, in the rectangle :math:`|x| < w_0, |y| < w_1`, with
homogeneous Dirichlet boundary conditions, the modes are products of a
cosine in each direction,

.. math::
    \exp \left\{
      -\frac{\kappa\pi^2 t}{4}
      \left(
        \frac{2n_0 + 1}{w_0^2} + \frac{2n_1 + 1}{w_1^2}
      \right)
    \right\}
    \cos\frac{\pi x}{2w_0}\cos\frac{\pi y}{2w_1}
for :math:`n_0, n_1 = 0, 1, 2, \ldots`.

Here we simulate the decay of the fundamental, :math:`n_0 = n_1 = 0`,
discretizing time using the generalized ('theta method') trapezoidal
rule.

For a constant time-step, this leads to a linear algebraic problem at
each time with the same matrix but changing right-hand side.  This
motivates factoring the matrix; e.g. with `scipy.sparse.linalg.splu`.


"""
from math import ceil
from typing import Iterator, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import splu

from skfem import *
from skfem.models.poisson import laplace, mass


halfwidth = np.array([2., 3.])
ncells = 2**3
diffusivity = 5.

mesh = MeshQuad.init_tensor(
    np.linspace(-1, 1, 2 * ncells) * halfwidth[0],
    np.linspace(-1, 1, 2 * ncells * ceil(halfwidth[1] //
                                         halfwidth[0])) * halfwidth[1])

element = ElementQuad1()
basis = InteriorBasis(mesh, element)

L = diffusivity * asm(laplace, basis)
M = asm(mass, basis)

dt = .01
print(f'dt = {dt} µs')
theta = 0.5                     # Crank–Nicolson
A = M + theta * L * dt
B = M - (1 - theta) * L * dt

boundary = basis.find_dofs()
interior = basis.complement_dofs(boundary)

# transpose as splu prefers CSC
backsolve = splu(condense(A, D=boundary, expand=False).T).solve

u_init = (np.cos(np.pi * mesh.p[0, :] / 2 / halfwidth[0])
          * np.cos(np.pi * mesh.p[1, :] / 2 / halfwidth[1]))


def step(t: float,
         u: np.ndarray) -> Tuple[float, np.ndarray]:
    u_new = np.zeros_like(u)              # zero Dirichlet conditions
    _, b1 = condense(csr_matrix(A.shape),  # ignore condensed matrix
                     B @ u, u_new, D=boundary, expand=False)
    u_new[interior] = backsolve(b1)
    return t + dt, u_new


def evolve(t: float,
           u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:

    while np.linalg.norm(u, np.inf) > 2**-3:
        t, u = step(t, u)
        yield t, u


if __name__ == '__main__':

    from argparse import ArgumentParser
    from pathlib import Path

    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    from skfem.visuals.matplotlib import plot

    parser = ArgumentParser(description='heat equation in a rectangle')
    parser.add_argument('-g', '--gif', action='store_true',
                        help='write animated GIF', )
    args = parser.parse_args()

    ax = plot(mesh, u_init, shading='gouraud')
    title = ax.set_title('t = 0.00')
    field = ax.get_children()[0]  # vertex-based temperature-colour
    fig = ax.get_figure()
    fig.colorbar(field)

    def update(event):
        t, u = event

        u0 = {'skfem': basis.interpolator(u)(np.zeros((2, 1)))[0],
              'exact': np.exp(-diffusivity * np.pi**2 * t / 4 *
                              sum(halfwidth**-2))}
        print('{:4.2f}, {:5.3f}, {:+7.4f}'.format(
            t, u0['skfem'], u0['skfem'] - u0['exact']))

        title.set_text(f'$t$ = {t:.2f}')
        field.set_array(u)

    animation = FuncAnimation(fig, update, evolve(0., u_init), repeat=False)
    if args.gif:
        animation.save(Path(__file__).with_suffix('.gif'), 'imagemagick')
    else:
        plt.show()
