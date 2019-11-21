from math import ceil
from typing import Iterator, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from sksparse.cholmod import cholesky

from skfem import *
from skfem.models.poisson import laplace, mass


halfwidth = np.array([2., 3.])
ncells = 2**3
diffusivity = 5.

mesh = MeshTri.init_tensor(
    np.linspace(-1, 1, 2 * ncells) * halfwidth[0],
    np.linspace(-1, 1, 2 * ncells * ceil(halfwidth[1] //
                                         halfwidth[0])) * halfwidth[1])

element = ElementTriP1()
basis = InteriorBasis(mesh, element)

L = diffusivity * asm(laplace, basis)
M = asm(mass, basis)

dt = .01
print(f'dt = {dt} µs')
theta = 0.5                     # Crank–Nicolson
A = M + theta * L * dt
B = M - (1 - theta) * L * dt

boundary = basis.get_dofs().all()
interior = basis.complement_dofs(boundary)

backsolve = cholesky(condense(A, D=boundary, expand=False)
                     .T)  # cholesky prefers CSC

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

    parser = ArgumentParser(description='heat equation in a rectangle')
    parser.add_argument('-g', '--gif', action='store_true',
                        help='write animated GIF', )
    args = parser.parse_args()

    ax = mesh.plot(u_init, smooth=True)  # smooth is vertex-based

    text = ax.text(.5, .9, 't = 0.',
                   bbox={'color': 'w'}, transform=ax.transAxes)

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

        text.set_text(f'$t$ = {t:.2f}')
        field.set_array(u)
        return text, field

    animation = FuncAnimation(fig, update, evolve(0., u_init),
                              blit=False, repeat=False)
    if args.gif:
        animation.save(Path(__file__).with_suffix('.gif'), 'imagemagick')
    else:
        plt.show()
