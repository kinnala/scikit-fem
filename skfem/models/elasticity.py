"""Linear elasticity."""

from skfem.assembly import bilinear_form


def lame_parameters(E, nu):
    return E / (2. * (1. + nu)), E * nu / ((1. + nu) * (1. - 2. * nu))


def linear_elasticity(Lambda=1., Mu=1.):
    @bilinear_form
    def weakform(u, du, v, dv, w):
        from .helpers import ddot, trace, transpose, eye
        
        def C(T):
            return 2.0 * Mu * T + Lambda * eye(trace(T), T.shape[0])

        def Eps(dw):
            return 0.5 * (dw + transpose(dw))

        return ddot(C(Eps(du)), Eps(dv))

    return weakform
