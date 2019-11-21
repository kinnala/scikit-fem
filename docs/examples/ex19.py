from math import ceil

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
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

backsolve = cholesky(condense(A, D=boundary, expand=False)
                     .T)  # cholesky prefers CSC


if __name__ == '__main__':

    ax = mesh.plot(u, smooth=True)
    field = ax.get_children()[0]
    fig = ax.get_figure()
    fig.colorbar(field)

    def evolve():
        t = 0.
        while np.linalg.norm(u) > 2**-4:
            _, b1 = condense(csr_matrix(A.shape),  # ignore condensed matrix
                             B @ u, D=boundary, expand=False)
            u[interior] = backsolve(b1)
            t += dt
            yield t, u

    def update(event):
        _, u = event
        field.set_array(u)
        return field,

    FuncAnimation(fig, update, evolve, blit=True, repeat=False)
    plt.show()
