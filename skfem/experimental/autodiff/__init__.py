from skfem.assembly.form.form import Form, FormExtraParams
from skfem.assembly.form.coo_data import COOData
from numpy import ndarray
import numpy as np
from jax import jvp, linearize, config
from jax.tree_util import register_pytree_node
import jax.numpy as jnp


config.update("jax_enable_x64", True)


class JaxDiscreteField(object):

    def __init__(self,
                 value,
                 grad=None,
                 div=None,
                 curl=None,
                 hess=None,
                 grad3=None,
                 grad4=None,
                 grad5=None,
                 grad6=None):
        self.value = value
        self.grad = grad
        self.div = div
        self.curl = curl
        self.hess = hess
        self.grad3 = grad3
        self.grad4 = grad4
        self.grad5 = grad5
        self.grad6 = grad6

    def __add__(self, other):
        if isinstance(other, JaxDiscreteField):
            return self.value + other.value
        return self.value + other

    def __sub__(self, other):
        if isinstance(other, JaxDiscreteField):
            return self.value - other.value
        return self.value - other

    def __mul__(self, other):
        if isinstance(other, JaxDiscreteField):
            return self.value * other.value
        return self.value * other

    def __rmul__(self, other):
        if isinstance(other, JaxDiscreteField):
            return self.value * other.value
        return self.value * other

    def __pow__(self, ix):
        return self.value ** ix

    def __array__(self):
        return self.value

    def __getitem__(self, index):
        return self.value[index]

    @property
    def shape(self):
        return self.value.shape

    @property
    def astuple(self):
        return (
            self.value,
            self.grad,
            self.div,
            self.curl,
            self.hess,
            self.grad3,
            self.grad4,
            self.grad5,
            self.grad6,
        )


register_pytree_node(
    JaxDiscreteField,
    lambda xs: (xs.astuple, None),
    lambda _, xs: JaxDiscreteField(*xs),
)


class NonlinearForm(Form):

    def assemble(self, basis, x=None, **kwargs):
        """Assemble the Jacobian and the right-hand side.

        Parameters
        ----------
        basis
            The basis used for all variables.
        x
            Optional point at which the form is linearized, default is zero.

        """
        # make x compatible with u in forms
        if x is None:
            x = basis.zeros()
        if isinstance(x, ndarray):
            x = basis.interpolate(x)
            if isinstance(x, tuple):
                x = tuple(JaxDiscreteField(*c.astuple) for c in x)
            else:
                x = (JaxDiscreteField(*x.astuple),)

        nt = basis.nelems
        dx = basis.dx

        defaults = basis.default_parameters()
        # turn defaults into JaxDiscreteField to avoid np.ndarray
        # to jnp.ndarray promotion issues
        w = FormExtraParams({
            **{
                k: JaxDiscreteField(*tuple(
                    jnp.asarray(x)
                    if x is not None else None
                    for x in defaults[k].astuple
                ))
                for k in defaults
            },
            **self._normalize_asm_kwargs(kwargs, basis),
        })

        # initialize COO data structures
        sz = basis.Nbfun * basis.Nbfun * nt
        data = np.zeros((basis.Nbfun, basis.Nbfun, nt), dtype=self.dtype)
        rows = np.zeros(sz, dtype=np.int32)
        cols = np.zeros(sz, dtype=np.int32)

        sz1 = basis.Nbfun * nt
        data1 = np.zeros(sz1, dtype=self.dtype)
        rows1 = np.zeros(sz1, dtype=np.int32)

        # # autograd version
        # def _make_jacobian(V):
        #     if 'hessian' in self.params:
        #         F = make_jvp(lambda U: self.form(*U, w))
        #         return make_jvp(lambda W: F(W)(V)[1])(x)
        #     return make_jvp(lambda U: self.form(*U, *V, w))(x)

        # JAX version
        def _make_jacobian(V):
            if 'hessian' in self.params:
                return linearize(
                    lambda W: jvp(lambda U: self.form(*U, w), (W,), (V,))[1],
                    x
                )
            return linearize(lambda U: self.form(*U, *V, w), x)

        # loop over the indices of local stiffness matrix
        for i in range(basis.Nbfun):
            y, DF = _make_jacobian(tuple(JaxDiscreteField(*c.astuple)
                                         for c in basis.basis[i]))
            for j in range(basis.Nbfun):
                DFU = DF(tuple(JaxDiscreteField(*c.astuple)
                               for c in basis.basis[j]))
                # Jacobian
                ixs = slice(nt * (basis.Nbfun * j + i),
                            nt * (basis.Nbfun * j + i + 1))
                rows[ixs] = basis.element_dofs[i]
                cols[ixs] = basis.element_dofs[j]
                data[j, i, :] = np.sum(DFU * dx, axis=1)
            # rhs
            ixs1 = slice(nt * i, nt * (i + 1))
            rows1[ixs1] = basis.element_dofs[i]
            data1[ixs1] = np.sum(y * dx, axis=1)

        data = data.flatten('C')

        return (
            COOData._assemble_scipy_csr(
                np.array([rows, cols]),
                data,
                (basis.N, basis.N),
                (basis.Nbfun, basis.Nbfun),
            ),
            COOData(
                np.array([rows1]),
                -data1,
                (basis.N,),
                (basis.Nbfun,)
            ).todefault()
        )
