r"""High-order plotting.

This simple example demonstrates the usage and visualisation of biquadratic
finite element basis. Many plotting tools, including matplotlib, provide tools
for visualising piecewise-linear triangular fields. Visualisation of
higher-order basis functions cannot be done directly and the mesh should be
further refined just for visualisation purposes.

:class:`~skfem.assembly.CellBasis` object includes the method
:meth:`~skfem.assembly.CellBasis.refinterp` which refines and simultaneously
interpolates any solution vector. The resulting mesh is non-conforming,
i.e. the connectivity between neighboring elements is lost, and hence it can
be used only for visualisation purposes.

.. note::

   As of 0.4.0, this functionality is included in :func:`~skfem.visuals.matplotlib.plot`,
   i.e. inputting :class:`~skfem.assembly.CellBasis` instead of :class:`~skfem.mesh.Mesh`
   uses :meth:`~skfem.assembly.CellBasis.refinterp` automatically.
   The steps in this example are still useful when, e.g., exporting to different
   formats for visualization purposes.

As an example, we solve the Poisson equation in a unit square with zero boundary
conditions and biquadratic basis on quadrilateral elements. The quadrilateral
elements are defined using an isoparametric local-to-global mapping.

"""

from skfem import *
from skfem.models.poisson import laplace, unit_load

m = MeshQuad().refined(2)

e1 = ElementQuad1()
e = ElementQuad2()
mapping = MappingIsoparametric(m, e1)
ib = CellBasis(m, e, mapping, 4)

K = asm(laplace, ib)

f = asm(unit_load, ib)

x = solve(*condense(K, f, D=ib.find_dofs()))

M, X = ib.refinterp(x, 3)

if __name__ == "__main__":
    from os.path import splitext
    from sys import argv
    from skfem.visuals.matplotlib import *
    ax = draw(m)
    plot(M, X, ax=ax, shading='gouraud', edgecolors='')
    savefig(splitext(argv[0])[0] + '_solution.png')
