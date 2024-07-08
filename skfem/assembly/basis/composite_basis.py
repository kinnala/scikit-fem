import numpy as np

from .abstract_basis import AbstractBasis
from skfem.element import ElementComposite


class CompositeBasis(AbstractBasis):
    """A combination of two or more Basis objects."""

    def __init__(self, *bases, equal_dofnum=False):

        nqp = len(bases[0].W)
        for basis in bases:
            if len(basis.W) != nqp:
                raise ValueError("Each Basis must have the same "
                                 "number of quadrature points.")
            if isinstance(basis.elem, ElementComposite):
                raise NotImplementedError("ElementComposite not "
                                          "supported.")

        self.X = bases[0].X
        self.W = bases[0].W
        self.bases = bases
        self.equal_dofnum = equal_dofnum
        self.nelems = bases[0].nelems
        self.dx = bases[0].dx
        self.default_parameters = bases[0].default_parameters

    @property
    def element_dofs(self):

        dofs = []
        offset = 0
        for basis in self.bases:
            dofs.append(basis.element_dofs + offset)
            if not self.equal_dofnum:
                offset += basis.N

        return np.vstack(dofs)

    @property
    def basis(self):

        bases = []
        M = len(self.bases)

        for i in range(M):
            for j in range(len(self.bases[i].basis)):
                tmp = []
                for k in range(M):
                    if k == i:
                        tmp.append(self.bases[i].basis[j][0])
                    else:
                        tmp.append(self.bases[i].basis[j][0].zeros())
                bases.append(tuple(tmp))

        return bases

    @property
    def N(self):
        if self.equal_dofnum:
            return self.bases[0].N

        # calculate sum of all N
        N = 0
        for basis in self.bases:
            N += basis.N
        return N

    @property
    def Nbfun(self):
        Nbfun = 0
        for basis in self.bases:
            Nbfun += basis.Nbfun
        return Nbfun

    def split(self, x):
        return list(zip(
            np.split(x, np.cumsum([basis.N
                                   for basis in self.bases])[:-1]),
            self.bases,
        ))
