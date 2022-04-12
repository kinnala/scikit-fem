from skfem import DiscreteField, BilinearForm, LinearForm
from skfem.assembly.form import Form
from jax import jvp
from jax.config import config


config.update("jax_enable_x64", True)


class NonlinearForm(Form):

    def linearize(self, u0):

        if not isinstance(u0, DiscreteField):
            raise NotImplementedError("NonlinearForm.linearize requires "
                                      "the point around which the form is "
                                      "linearized as an argument.")

        @BilinearForm
        def DF(u, v, w):
            F1 = lambda U: self.form(U, v, w)
            return jvp(F1, (u0.astuple,), (u.astuple,))[1]

        @LinearForm
        def y(v, w):
            return -self.form(u0.astuple, v, w)

        return DF, y

    def hessian(self, u0):
        pass
