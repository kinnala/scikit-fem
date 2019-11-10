from ..element_h1 import ElementH1


class ElementTriDG(ElementH1):
    dim = 2

    def __init__(self, elem):
        # change all dofs to interior dofs
        self.elem = elem
        self.maxdeg = elem.maxdeg
        self.interior_dofs = (3*elem.nodal_dofs +
                              3*elem.facet_dofs +
                              elem.interior_dofs)

    def lbasis(self, X, i):
        return self.elem.lbasis(X, i)
