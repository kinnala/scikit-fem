"""Weak forms for elasticity."""

from skfem.assembly import BilinearForm
from skfem.helpers import ddot, trace, sym_grad, eye


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
        The first Lamé parameter (lambda)
    float
        The second Lamé parameter (mu)

    """
    return (E * nu / ((1. + nu) * (1. - 2. * nu)),
            E / (2. * (1. + nu)))


def plane_stress(E, nu):
    """Map Young's modulus and Poisson ratio to plane stress lame parameters.
    Parameters
    ----------
    E
        Young's modulus
    nu
        Poisson ratio

    Returns
    -------
    lam: float
        The first Lamé parameter for plane stress scenario (lambda)
    mu: float
        The second Lamé parameter for plane stress scenario (mu)
    """
    lam = E * nu / (1. - nu**2)
    mu = E / (2. * (1. + nu))
    return (lam, mu)


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
