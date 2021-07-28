r"""Heat equation in one-dimension.

The solution of the heat equation is reduced from two dimensions in ex19 to
one here, to demonstrate the different post-processing required.

The two-dimensional modes from ex19 reduce in the limit
:math:`w_1 \rightarrow\infty` of the strip :math:`|x| < w` to

.. math::
    \exp \left[
      -\kappa
      \left\{
        \left(\frac{(2n + 1)\pi}{2w}\right)^2
      \right\}
      t
    \right]
    \cos\frac{(2n + 1)\pi x}{2w}
for :math:`n = 0, 1, 2, \ldots`.

Here we simulate the decay of the fundamental, :math:`n = 0`, again
discretizing time using the generalized ('theta method') trapezoidal
rule.

"""
from typing import Iterator, Tuple

import numpy as np
from scipy.sparse.linalg import splu

from skfem import *
from skfem.models.poisson import laplace, mass


halfwidth = 2.0
ncells = 2 ** 3
diffusivity = 5.0

mesh = MeshLine(np.linspace(-1, 1, 2 * ncells) * halfwidth)

element = ElementLineP2()  # or ElementLineP1
basis = Basis(mesh, element)

L = diffusivity * asm(laplace, basis)
M = asm(mass, basis)

dt = 0.01
print("dt =", dt)
theta = 0.5  # Crank–Nicolson
L0, M0 = penalize(L, M, D=basis.find_dofs())
A = M0 + theta * L0 * dt
B = M0 - (1 - theta) * L0 * dt

backsolve = splu(A.T).solve  # .T as splu prefers CSC

u_init = np.cos(np.pi * basis.doflocs[0] / 2 / halfwidth)


def evolve(t: float, u: np.ndarray) -> Iterator[Tuple[float, np.ndarray]]:

    while np.linalg.norm(u, np.inf) > 2 ** -3:
        t, u = t + dt, backsolve(B @ u)
        yield t, u


if __name__ == "__main__":

    from argparse import ArgumentParser
    from pathlib import Path

    from matplotlib.animation import FuncAnimation
    import matplotlib.pyplot as plt

    parser = ArgumentParser(description="heat equation in a slab")
    parser.add_argument(
        "-g", "--gif", action="store_true", help="write animated GIF",
    )
    args = parser.parse_args()

    sorting = np.argsort(basis.doflocs)[0]
    fig, ax = plt.subplots()
    ax.set_xlabel(r"$x/w$")
    ax.set_ylabel(r"reduced temperature, $u(x, t)/u(0, 0)$")
    ax.set_xlim(-halfwidth, halfwidth)
    ax.set_ylim(0, 1)
    title = ax.set_title("t = 0.00")
    line = ax.plot(basis.doflocs[0, sorting], u_init[sorting], marker="o")[0]

    probe = basis.probes(np.zeros((mesh.dim(), 1)))

    def update(event):
        t, u = event

        u0 = {
            "skfem": (probe @ u)[0],
            "exact": np.exp(-diffusivity * (np.pi / 2 / halfwidth) ** 2 * t),
        }
        print(
            "{:4.2f}, {:5.3f}, {:+7.4f}".format(
                t, u0["skfem"], u0["skfem"] - u0["exact"]
            )
        )

        title.set_text(f"$t$ = {t:.2f}")
        line.set_ydata(u[sorting])

    animation = FuncAnimation(fig, update, evolve(0.0, u_init), repeat=False)
    if args.gif:
        animation.save(Path(__file__).with_suffix(".gif"), "imagemagick")
    else:
        plt.show()
