from ..element_h1 import ElementH1


class ElementTetP0(ElementH1):
    interior_dofs = 1
    dim = 3
    maxdeg = 0
    dofnames = ['u']

    def lbasis(self, X, i):
        return 1 + 0*X[0, :], 0*X
