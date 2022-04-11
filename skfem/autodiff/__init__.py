from skfem import DiscreteField, BilinearForm, LinearForm
from skfem.assembly.form import Form
from jax import linearize, jvp
from jax.config import config


config.update("jax_enable_x64", True)


def derivative(fun, u0):
    if not isinstance(u0, DiscreteField):
        raise NotImplementedError("Second argument must be DiscreteField.")
    y, DF = linearize(fun, u0.astuple)
    return y, lambda u: DF(u.astuple)


def diff(F, u0):
    if not isinstance(u0, DiscreteField):
        raise NotImplementedError("Second argument must be DiscreteField.")

    @BilinearForm
    def DF(u, v, w):
        F1 = lambda U: F(U, v, w)
        return jvp(F1, (u0.astuple,), (u.astuple,))[1]

    @LinearForm
    def y(v, w):
        F1 = lambda U: F(U, v, w)
        return -F1(u0.astuple)

    return y, DF
