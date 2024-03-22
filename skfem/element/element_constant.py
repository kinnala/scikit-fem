import numpy as np

from .element_h1 import ElementH1
from ..refdom import RefTri
from .discrete_field import DiscreteField


class ElementConstant(ElementH1):

    global_dofs = 1
    maxdeg = 1
    dofnames = ['u']
    refdom = RefTri
    doflocs = [[np.nan, np.nan]]

    def __init__(self, value=None, grad=None):
        if value is None:
            self.value = lambda x: np.ones_like(x[0])
        else:
            self.value = value
        if grad is None:
            self.grad = np.zeros_like
        else:
            self.grad = grad

    def gbasis(self, mapping, X, i, tind=None):
        x = mapping.F(X, tind)
        return (DiscreteField(
            value=self.value(x),
            grad=self.grad(x)
        ),)
