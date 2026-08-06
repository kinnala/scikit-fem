r"""Matérn covariance as a sparse precision matrix.

A Gaussian field with the Matérn covariance is the stationary solution of the
stochastic partial differential equation

.. math::
   (\kappa^2 - \Delta) u = \mathcal{W}

driven by Gaussian white noise :math:`\mathcal{W}` [WHITTLE]_.  In two
dimensions this is the case :math:`\alpha = 2` of the family, so the smoothness
is :math:`\nu = 1` and the correlation range is :math:`\sqrt 8/\kappa`.
Discretizing the equation with linear triangles gives a Gaussian Markov random
field with the sparse precision matrix

.. math::
   Q = (\kappa^2 C + G)\,C^{-1}(\kappa^2 C + G),

where :math:`C` is the lumped mass matrix and :math:`G` is the stiffness matrix
[LINDGREN]_.  Lumping the mass is what keeps :math:`Q` sparse; with the
consistent mass matrix the middle factor is full.

The three panels are correlations with respect to one node, formed from a
column and the diagonal of :math:`Q^{-1}`.  The first has a constant
:math:`\kappa` and is compared with the analytical Matérn correlation
:math:`\kappa r K_1(\kappa r)`; the discrepancy is dominated by the natural
boundary condition of the weak form, which reflects correlation back into the
square.  The second gives :math:`\kappa` a different value on each half of the
domain.  The third puts a wall of nodes of very short range across the middle,
leaving gaps at the top and the bottom, and the correlation is routed around
its ends [BAKKA]_.

.. [WHITTLE] Whittle, P. (1963). Stochastic processes in several dimensions. *Bulletin of the International Statistical Institute* 40, 974–985.

.. [LINDGREN] Lindgren, F., Rue, H. & Lindström, J. (2011). An explicit link between Gaussian fields and Gaussian Markov random fields: the stochastic partial differential equation approach. *Journal of the Royal Statistical Society: Series B* 73(4), 423–498. `doi:10.1111/j.1467-9868.2011.00777.x <https://doi.org/10.1111/j.1467-9868.2011.00777.x>`_

.. [BAKKA] Bakka, H., Vanhatalo, J., Illian, J. B., Simpson, D. & Rue, H. (2019). Non-stationary Gaussian models with physical barriers. *Spatial Statistics* 29, 268–288. `doi:10.1016/j.spasta.2019.01.002 <https://doi.org/10.1016/j.spasta.2019.01.002>`_

"""
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import splu
from scipy.special import kv

from skfem import *
from skfem.models.poisson import laplace, mass

# refined towards the centre, where the correlations are evaluated
mesh = MeshTri().refined(2)
for _ in range(3):
    cx, cy = mesh.p[:, mesh.t].mean(1)
    mesh = mesh.refined(np.nonzero(np.hypot(cx - .5, cy - .5) < .45)[0])

basis = Basis(mesh, ElementTriP1())
x, y = basis.doflocs
N = basis.N

C = np.asarray(mass.assemble(basis).sum(1)).ravel()  # lumped, so diagonal
G = laplace.assemble(basis)


def kappa_of(rho):
    """the kappa corresponding to a correlation range"""
    return np.sqrt(8.) / rho


def factorize(kappa):
    """factorize A of Q = A diag(1 / C) A; kappa is a scalar or nodewise"""
    return splu((diags(kappa ** 2 * C) + G).tocsc())


def variances(lu):
    """the diagonal of the covariance Q^-1 = A^-1 diag(C) A^-1"""
    return (lu.solve(np.diag(np.sqrt(C))) ** 2).sum(1)


def correlation(lu, var, i):
    """the correlation of the field with node i"""
    e = np.zeros(N)
    e[i] = 1.
    return lu.solve(C * lu.solve(e)) / np.sqrt(var[i] * var)


centre = np.argmin((x - .5) ** 2 + (y - .5) ** 2)
rho = .55  # the correlation range

lu = factorize(kappa_of(rho))
var = variances(lu)
corr = correlation(lu, var, centre)

# the marginal variance of the continuous field is 1 / (4 pi kappa^2)
var_ratio = 4. * np.pi * kappa_of(rho) ** 2 * var[centre]

r = np.hypot(x - x[centre], y - y[centre])
kr = kappa_of(rho) * r
matern = np.where(kr > 0, kr * kv(1, np.where(kr > 0, kr, 1.)), 1.)
band = (r > .05) & (r < .35)  # r = 0 is unity by construction
err = np.abs(corr[band] - matern[band]).max()
print('max. error against the analytical correlation = {:.4f}'.format(err))
print('marginal variance / analytical = {:.4f}'.format(var_ratio))

# a long range on the left of the domain and a short one on the right
lu_ns = factorize(kappa_of(np.where(x < .5, .9, .25)))
corr_ns = correlation(lu_ns, variances(lu_ns), centre)
# probes placed symmetrically about the centre
left = np.argmin((x - .25) ** 2 + (y - .5) ** 2)
right = np.argmin((x - .75) ** 2 + (y - .5) ** 2)
lopsided = corr_ns[left] / corr_ns[right]
print('correlation on the long side / the short = {:.4f}'.format(lopsided))

# a wall of very short range, with gaps left open at the top and the bottom
kappa_bar = np.full(N, kappa_of(rho))
wall = (np.abs(x - .5) < .05) & (np.abs(y - .5) < .18)
kappa_bar[wall] = kappa_of(.01)
source = np.argmin((x - .35) ** 2 + (y - .5) ** 2)  # to the left of the wall
across = np.argmin((x - .65) ** 2 + (y - .5) ** 2)  # straight behind it
beyond = np.argmin((x - .65) ** 2 + (y - .2) ** 2)  # past its lower end
lu_bar = factorize(kappa_bar)
corr_bar = correlation(lu_bar, variances(lu_bar), source)
corr_free = correlation(lu, var, source)  # the same source, without the wall
blocked = corr_bar[across] / corr_free[across]
routed = corr_bar[beyond] / corr_bar[across]
print('behind the wall {:.4f} of the unobstructed correlation, and {:.4f}'
      ' times that past its end'.format(blocked, routed))


def visualize():
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize
    from skfem.visuals.matplotlib import draw, plot
    fig, ax = plt.subplots(1, 3, figsize=(12, 4.6), layout='constrained')
    for a, cvals, i, title in [(ax[0], corr, centre, 'Constant range'),
                               (ax[1], corr_ns, centre, 'Varying range'),
                               (ax[2], corr_bar, source, 'Barrier')]:
        plot(mesh, cvals, ax=a, shading='gouraud', cmap='magma',
             vmin=0., vmax=1.)
        draw(mesh, ax=a, linewidth=.2)  # the mesh, penciled over the field
        a.plot(x[i], y[i], 'o', ms=6, color='w', markeredgecolor='k',
               markeredgewidth=.8, zorder=7)
        a.set_title(title)
        a.set_aspect('equal')
        a.set_axis_off()
    ax[2].plot(x[wall], y[wall], '.', ms=4, color='c', zorder=6)  # the wall
    fig.colorbar(ScalarMappable(Normalize(0., 1.), 'magma'), ax=ax, shrink=.85,
                 label='correlation')
    return ax[0]


if __name__ == "__main__":
    visualize().show()
