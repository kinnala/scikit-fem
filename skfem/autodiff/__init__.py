import logging

from skfem import DiscreteField, BilinearForm, LinearForm
from skfem.assembly.form import Form
from jax import jvp
from jax.config import config
from numpy import ndarray


config.update("jax_enable_x64", True)
logger = logging.getLogger(__name__)
logger.warning("skfem.autodiff is an experimental feature which requires JAX "
               "for automatic differentiation of forms. You can install JAX "
               "by running 'pip install jax[cpu]'.")


class NonlinearForm(Form):

    def linearize(self, u0: DiscreteField):

        if not isinstance(u0, DiscreteField):
            raise NotImplementedError

        @BilinearForm
        def DF(u, v, w):
            return jvp(lambda U: self.form(U, v, w),
                       (u0.astuple,),
                       (u.astuple,))[1]

        @LinearForm
        def y(v, w):
            return self.form(u0.astuple, v, w)

        return DF, y

    def assemble(self, u0, basis):

        if isinstance(u0, ndarray):
            u0 = basis.interpolate(u0)

        DF, y = self.linearize(u0)

        return (
            DF.assemble(basis),
            y.assemble(basis),
        )


class NonlinearFunctional(Form):

    def linearize(self, u0):

        if not isinstance(u0, DiscreteField):
            raise NotImplementedError

        @BilinearForm
        def HF(u, v, w):
            return jvp(lambda U0: jvp(lambda U: self.form(U, w),
                                      (U0,),
                                      (v.astuple,))[1],
                       (u0.astuple,),
                       (u.astuple,))[1]

        @LinearForm
        def DF(v, w):
            return jvp(lambda U: self.form(U, w),
                       (u0.astuple,),
                       (v.astuple,))[1]

        return HF, DF

    def assemble(self, u0, basis):

        if isinstance(u0, ndarray):
            u0 = basis.interpolate(u0)

        HF, DF = self.linearize(u0)

        return (
            HF.assemble(basis),
            DF.assemble(basis),
        )
