r"""# Linear hydrodynamic stability

The linear stability of one-dimensional solutions of the Navier–Stokes equations
is governed by the `Orr–Sommerfeld equation
<https://en.wikipedia.org/wiki/Orr%E2%80%93Sommerfeld_equation>`_ (Drazin &
Reid 2004, p. 156).  This is expressed in terms of the stream-function
:math:`\phi` of the perturbation, giving a two-point boundary value problem

.. math::
   \alpha\phi(\pm 1) = \phi'(\pm 1) = 0
for a complex fourth-order ordinary differential equation,

.. math::
   \left(\alpha^2-\frac{\mathrm d^2}{\mathrm dz^2}\right)^2\phi
   = (\mathrm j\alpha R)\left\{
     (c - U)\left(\alpha^2-\frac{\mathrm d^2}{\mathrm dz^2}\right)\phi
     - U''\phi,
   \right\}

where :math:`U(z)` is the base velocity profile, :math:`c` and :math:`\alpha`
are the wavespeed and wavenumber of the disturbance, and :math:`R` is the
Reynolds number.  In the 'temporal' stability problem, :math:`R` and
:math:`\alpha` are specified as positive and :math:`c` found as the complex
eigenvalue.

The fourth-order derivatives would require :math:`C^1` finite elements, e.g. of
the Hermite family (Mamou & Khalid 2004); however, this can be avoided by
reverting to the system from which the Orr–Sommerfeld stream-function equation
is derived (Drazin & Reid 2004, eq. 25.9, p. 155), which is expressed in terms
of pressure :math:`p` and longitudinal :math:`u` and transverse :math:`w`
components of the disturbance to the velocity:

.. math::
   
   \left(\mathrm j\alpha R U(z) + \alpha^2 - \frac{\mathrm d^2}{\mathrm dz^2}\right)u + RU'(z)w + \mathrm j\alpha  Rp &= \mathrm j\alpha Rc u \\
   \left(\mathrm j\alpha R U(z) + \alpha^2 - \frac{\mathrm d^2}{\mathrm dz^2}\right)w + R\frac{\mathrm dp}{\mathrm dz} &= \mathrm j\alpha Rc w \\
   \mathrm j\alpha R u + R\;\frac{\mathrm dw}{\mathrm dz} &= 0
   
This primitive system is second-order and can be discretized using
one-dimensional Taylor–Hood or Mini elements; here the former are the default with the latter selectable with the `--element Mini` command-line option.

The classical test-case for this problem is plane Poiseuille flow :math:`U(z) =
1 - z^2` on :math:`-1 < z < 1` at :math:`\alpha = 1` and :math:`R = 10^4`
(Drazin & Reid 2004, figure 4.19; Mamou & Khalid 2004), typically seeking only
the 'even' modes for which :math:`u(0) = u(1) = w'(0) = w(1) = 0`.  Good
agreement with reference results for the complex wavespeed spectrum (Criminale,
Jackson, & Joslin 2003, table 3.1) is obtained on a uniform mesh of 64
segments.

* Criminale, W. O., Jackson, T. L.,, Joslin, R. D. (2003). *Theory and Computation in Hydrodynamic Stability.* Cambridge: Cambridge University Press. `doi:10.1017/CBO9780511550317 <https://doi.org/10.1017%2fCBO9780511550317>`_
* Drazin, P. G., Reid, W. H. (2004). *Hydrodynamic Stability.* Cambridge University Press. `doi:10.1017/CBO9780511616938 <https://doi.org/10.1017%2fCBO9780511616938>`_
* Mamou, M. & Khalid, M. (2004). Finite element solution of the Orr–Sommerfeld equation using high precision Hermite elements: plane Poiseuille flow. *International Journal for Numerical Methods in Fluids* 44. pp. 721–735. `doi:10.1002/fld.661 <https://doi.org/10.1002%2ffld.661>`_

"""
from skfem import *
import skfem.element.element_line as element_line
from skfem.models.general import divergence
from skfem.models.poisson import laplace, mass

from argparse import ArgumentParser
from pathlib import Path

from matplotlib.pyplot import subplots
import numpy as np
from numpy.polynomial.polynomial import Polynomial
from scipy.sparse import block_diag, bmat, csr_matrix
from scipy.sparse.linalg import eigs


if __name__ == '__main__':
    parser = ArgumentParser(description='Orr-Sommerfeld equation')
    parser.add_argument('-e', '--element', type=str, default='P2',
                        help='velocity element')
    args = parser.parse_args()
    u_element = args.element
else:
    u_element = 'P2'
    
U = Polynomial([1, 0, -1])      # base-flow profile


@BilinearForm
def base_velocity(u, v, w):
    return v * U(w.x[0]) * u


@BilinearForm
def base_shear(u, v, w):
    return v * U.deriv()(w.x[0]) * u


mesh = MeshLine(np.linspace(0, 1, 2**6)).with_boundaries(
    {
        "centre": lambda x: x[0] == 0,
        "wall": lambda x: x[0] == 1
    }
)
element = {'u': getattr(element_line, f'ElementLine{u_element}')(),
           'p': ElementLineP1()}
basis = {v: Basis(mesh, e, intorder=4) for v, e in element.items()}

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

walls = np.hstack([basis["u"].get_dofs(),
                   basis["u"].get_dofs("wall").all() + basis['u'].N])

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
