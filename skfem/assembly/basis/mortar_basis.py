import numpy as np
from numpy import ndarray

from skfem.quadrature import get_quadrature
from .basis import Basis
from .interior_basis import InteriorBasis


class MortarBasis(Basis):
    """Global basis functions evaluated at integration points on the mortar
    boundary. """

    def __init__(self,
                 mesh,
                 elem,
                 intorder: int = None,
                 side: int = 0):
        """Initialize a basis for assembling mortar matrices.

        Parameters
        ----------
        mesh
            An object of type :class:`~skfem.mesh.MeshMortar`.
        elem
            An object of type :class:`~skfem.element.Element`.
        intorder
            Required integration order, i.e. the degree of polynomials that are
            integrated exactly by the used quadrature.
        side
            The numbers 0 and 1 refer to the different sides of the mortar
            interface. Side 0 corresponds to the indices mesh.f2t[0, :].

        """
        if intorder is None:
            raise ValueError("Please specify 'intorder' for MortarBasis "
                             "and use consistent orders for both sides.")

        # build mapping for the mortar mesh
        #super(MortarBasis, self).__init__(mesh, elem, None, intorder)

        self.mesh = mesh.target_mesh[side]
        #self.brefdom = mesh.brefdom
        # self.refdom = mesh.brefdom
        # mesh.refdom = mesh.brefdom
        #self.intorder = 3
        #self.mapping = mesh.mapping()
        self.elem = elem

        # build dofnum and mappings for the target mesh
        self._build_dofnum(mesh.target_mesh[side], elem)
        integ_mapping = mesh.supermesh.mapping()
        target_mapping = mesh.target_mesh[side].mapping()
        helper_mapping = mesh.helper_mesh[side].mapping()
        self.Nbfun = self.element_dofs.shape[0]

        self.X, self.W = get_quadrature('line', 3)

        # self.find = np.arange(self.mesh.facets.shape[1])
        self.find = mesh.I[side]
        self.tind = mesh.target_mesh[side].f2t[0, mesh.I[side]]
        self.mapping = target_mapping

        # boundary refdom to global facet
        x = integ_mapping.F(self.X)
        X = helper_mapping.invF(x, tind=mesh.ix[side])
        x = target_mapping.G(X, find=self.find)

        # global facet to refdom facet
        Y = target_mapping.invF(x, tind=self.tind)

        # normals are defined in the mortar mesh
        self.normals = np.repeat(mesh.normals[:, :, None],
                                 len(self.W),
                                 axis=2)

        self.basis = [self.elem.gbasis(target_mapping, Y, j, self.tind)
                      for j in range(self.Nbfun)]

        self.nelems = len(self.find)
        self.dx = np.abs(self.mapping.detDG(X, find=self.find)) *\
            np.tile(self.W, (self.nelems, 1))

        self.element_dofs = self.element_dofs[:, self.tind]


    def default_parameters(self):
        """Return default parameters for `~skfem.assembly.asm`."""
        return {'x': self.global_coordinates(),
                'h': self.mesh_parameters(),
                'n': self.normals}
    
    def global_coordinates(self) -> ndarray:
        return self.mapping.G(self.X, find=self.find)

    def mesh_parameters(self) -> ndarray:
        if self.mesh.dim() == 1:
            return np.array([0.0])
        else:
            return np.abs(self.mapping.detDG(self.X, self.find)) ** (1.0 / (self.mesh.dim() - 1))
