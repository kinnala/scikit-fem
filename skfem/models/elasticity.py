"""Weak forms for elasticity."""

from skfem.assembly import BilinearForm
from .helpers import ddot, trace, sym_grad,\
    transpose, eye, grad


def lame_parameters(E, nu):
    """Calculate the Lamé parameters from E and nu.

    Parameters
    ----------
    E
        Young's modulus
    nu
        Poisson ratio

    Returns
    -------
    float
        The first Lamé parameter
    float
        The second Lamé parameter

    """
    return (E / (2. * (1. + nu)),
            E * nu / ((1. + nu) * (1. - 2. * nu)))


def linear_stress(Lambda=1., Mu=1.):

    def C(T):
        """Linear-elastic stress-strain relationship."""
        return 2. * Mu * T + Lambda * eye(trace(T), T.shape[0])

    return C


def linear_elasticity(Lambda=1., Mu=1.):
    """Weak form of the linear elasticity operator."""

    C = linear_stress(Lambda, Mu)

    @BilinearForm
    def weakform(u, v, w):
        return ddot(C(sym_grad(u)), sym_grad(v))

    return weakform
