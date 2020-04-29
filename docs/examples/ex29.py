from skfem import *
from skfem.models.general import divergence
from skfem.models.poisson import laplace, mass

from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.sparse import block_diag, bmat, csr_matrix
from scipy.sparse.linalg import eigs


U = Polynomial([1, 0, -1])      # base-flow profile


@BilinearForm
def base_velocity(u, v, w):
    return v * U(w.x[0]) * u


@BilinearForm
def base_shear(u, v, w):
    return v * U.deriv()(w.x[0]) * u


mesh = MeshLine(np.linspace(0, 1, 2**6))
element = {'u': ElementLineP2(), 'p': ElementLineP1()}
basis = {v: InteriorBasis(mesh, e, intorder=4) for v, e in element.items()}

L = asm(laplace, basis['u'])
M = asm(mass, basis['u'])
P = asm(mass, basis['u'], basis['p'])
B = -asm(divergence, basis['u'], basis['p'])
V = asm(base_velocity, basis['u'])
W = asm(base_shear, basis['u'])

re = 1e4                        # Reynolds number
alpha = 1.                      # longitudinal wavenumber
jare = 1j * alpha * re
Kuu = jare * V + alpha**2 * M + L
stiffness = bmat([[Kuu,       re * W, jare * P.T],
                  [None,      Kuu,    re * B.T],
                  [-jare * P, re * B, None]], 'csc')
mass_matrix = block_diag([M, M, csr_matrix((basis['p'].N,)*2)], 'csr')

# Seek only 'even' modes, 'even' in terms of conventional
# stream-function formulation, so that the longitudinal component u of
# the perturbation to the velocity vanishes on the centre-line z = 0,
# z here being the sole coordinate.

u_boundaries = basis['u'].find_dofs()['all'].all()
walls = np.concatenate([u_boundaries,
                        u_boundaries[1:] + basis['u'].N])

pencil = condense(stiffness, mass_matrix, D=walls, expand=False)
c = {'Criminale et al': np.loadtxt(Path(__file__).with_suffix('.csv'),
                                   dtype=complex)}
c['scikit-fem'] = eigs(pencil[0], M=pencil[1],
                       k=len(c['Criminale et al']), sigma=0.,
                       return_eigenvectors=False) / jare


if __name__ == '__main__':
    fig, ax = subplots()
    for (label, wavespeed), marker in zip(c.items(), 'o+'):
        ax.plot(wavespeed.real, wavespeed.imag,
                marker=marker, linestyle='None', label=label)
    fig.suptitle('even modes of plane Poiseuille flow (Re=10⁴, α=1)')
    ax.set_xlabel(r'real wavespeed, $\Re c$')
    ax.set_xlim((0, 1))
    ax.set_ylabel(r'imaginary wavespeed, $\Im c$')
    ax.set_ylim((-.8, .1))
    ax.grid(True)
    ax.legend(loc=3)
    fig.savefig(Path(__file__).with_suffix('.png'))
