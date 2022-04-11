from skfem import DiscreteField
from jax import linearize
from jax.config import config


config.update("jax_enable_x64", True)


def derivative(fun, u0):
    if not isinstance(u0, DiscreteField):
        raise NotImplementedError("Second argument must be DiscreteField.")
    y, DF = linearize(fun, u0.astuple)
    return y, lambda u: DF(u.astuple)
