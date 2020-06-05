from unittest import TestCase
from skfem import *
from skfem.mesh.topology import Topology
from skfem.mesh.geometry import Geometry
from skfem.assembly.dofnum import Dofnum
from skfem.models.poisson import laplace


class TestHighOrderMachinery(TestCase):

    def runTest(self):
        m = MeshTri()
        topo = Topology(m.t)
        dofnum = Dofnum(topo, ElementTriP1())
        geom = Geometry(m.p, dofnum)
        basis = InteriorBasis(geom, ElementTriP2())
        std_basis = InteriorBasis(m, ElementTriP2())

        self.assertEqual((laplace.assemble(basis)
                          - laplace.assemble(std_basis)).sum(), 0)
