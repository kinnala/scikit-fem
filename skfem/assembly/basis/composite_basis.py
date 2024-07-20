import numpy as np

from .abstract_basis import AbstractBasis
from skfem.element import ElementComposite


class CompositeBasis(AbstractBasis):
    """A combination of two or more Basis objects."""

    def __init__(self, *bases, equal_dofnum=False):

        nelem = bases[0].element_dofs.shape[1]
        nqp = len(bases[0].W)
        for basis in bases:
            if len(basis.W) != nqp:
                raise ValueError("Each Basis must have the same "
                                 "number of quadrature points.")
            if bases[0].element_dofs.shape[1] != nelem:
                raise ValueError("Each Basis must have the same "
                                 "number of elements.")
            if isinstance(basis.elem, ElementComposite):
                raise NotImplementedError("ElementComposite not "
                                          "supported.")

        self.bases = bases
        self.equal_dofnum = equal_dofnum

        # for caching
        self._element_dofs = None
        self._basis = None

    def default_parameters(self):
        return self.bases[0].default_parameters()

    @property
    def dx(self):
        return self.bases[0].dx

    @property
    def nelems(self):
        return self.bases[0].nelems

    @property
    def X(self):
        return self.bases[0].X

    @property
    def W(self):
        return self.bases[0].W

    @property
    def element_dofs(self):

        if self._element_dofs is None:
            dofs = []
            offset = 0
            for basis in self.bases:
                dofs.append(basis.element_dofs + offset)
                if not self.equal_dofnum:
                    offset += basis.N
            self._element_dofs = np.vstack(dofs)

        return self._element_dofs

    @property
    def basis(self):

        if self._basis is None:
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

            self._basis = bases

        return self._basis

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

    def __repr__(self):
        rep = ""
        rep += "<skfem CompositeBasis object>\n"
        rep += "  Number of DOFs: {}\n".format(self.N)
        rep += "  Consist of: {}\n".format(repr(self.bases))
        return rep

    def interpolate(self, x):

        # find slice indices
        ixs = [0]
        for basis in self.bases:
            ixs.append(basis.N + ixs[-1])

        return tuple(basis.interpolate(x[ixs[itr]:ixs[itr + 1]])
                     for itr, basis in enumerate(self.bases))

    def get_dofs(self, *args, **kwargs):
        raise NotImplementedError
