from skfem import DiscreteField, BilinearForm, LinearForm
from skfem.assembly.form import Form
from jax import jvp
from jax.config import config
from numpy import ndarray


config.update("jax_enable_x64", True)


class NonlinearForm(Form):

    def linearize(self, u0):

        if not isinstance(u0, DiscreteField):
            raise NotImplementedError("linearize requires the point around "
                                      "which the form is linearized as "
                                      "an argument.")

        @BilinearForm
        def DF(u, v, w):
            F1 = lambda U: self.form(U, v, w)
            return jvp(F1, (u0.astuple,), (u.astuple,))[1]

        @LinearForm
        def y(v, w):
            return -self.form(u0.astuple, v, w)

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

        @BilinearForm
        def HF(u, v, w):
            F1 = lambda U: self.form(U, w)
            F2 = lambda U: jvp(F1, (U,), (v.astuple,))[1]
            return jvp(F2, (u0.astuple,), (u.astuple,))[1]

        @LinearForm
        def DF(v, w):
            F1 = lambda U: self.form(U, w)
            return -jvp(F1, (u0.astuple,), (v.astuple,))[1]

        return HF, DF

    def assemble(self, u0, basis):

        if isinstance(u0, ndarray):
            u0 = basis.interpolate(u0)

        HF, DF = self.linearize(u0)

        return (
            HF.assemble(basis),
            DF.assemble(basis),
        )
