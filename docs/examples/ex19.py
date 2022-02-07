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
    \exp \left[
      -\frac{\kappa\pi^2 t}{4}
      \left\{
        \left(\frac{2n_0 + 1}{w_0}\right)^2 +
        \left(\frac{2n_1 + 1}{w_1}\right)^2
      \right\}
    \right]
    \cos\frac{(2n_0 + 1)\pi x}{2w_0}
    \cos\frac{(2n_1 + 1)\pi y}{2w_1}
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

element = ElementQuad2()  # or ElementQuad1
basis = Basis(mesh, element)

L = diffusivity * asm(laplace, basis)
M = asm(mass, basis)

dt = .01
print('dt =', dt)
theta = 0.5                     # Crank–Nicolson
L0, M0 = penalize(L, M, D=basis.get_dofs())
A = M0 + theta * L0 * dt
B = M0 - (1 - theta) * L0 * dt

backsolve = splu(A.T).solve  # .T as splu prefers CSC

u_init = np.cos(np.pi * basis.doflocs / 2 / halfwidth[:, None]).prod(0)


def exact(t: float) -> np.ndarray:
    return np.exp(-diffusivity * np.pi ** 2 * t / 4 * sum(halfwidth ** -2)) * u_init


def evolve(t: float, 
           u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:

    while np.linalg.norm(u, np.inf) > 2**-3:
        t, u = t + dt, backsolve(B @ u)
        yield t, u


probe = basis.probes(np.zeros((mesh.dim(), 1)))


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

    ax = plot(mesh, u_init[basis.nodal_dofs.flatten()], shading='gouraud')
    title = ax.set_title('t = 0.00')
    field = ax.get_children()[0]  # vertex-based temperature-colour
    fig = ax.get_figure()
    fig.colorbar(field)


    def update(event):
        t, u = event

        u0 = {'skfem': (probe @ u)[0], 
              'exact': (probe @ exact(t))[0]}
        print('{:4.2f}, {:5.3f}, {:+7.4f}'.format(
                t, u0['skfem'], u0['skfem'] - u0['exact']))

        title.set_text(f'$t$ = {t:.2f}')
        field.set_array(u[basis.nodal_dofs.flatten()])

    animation = FuncAnimation(
        fig,
        update,
        evolve(0., u_init),
        repeat=False,
        interval=50,
    )
    if args.gif:
        animation.save(Path(__file__).with_suffix('.gif'), 'imagemagick')
    else:
        plt.show()
