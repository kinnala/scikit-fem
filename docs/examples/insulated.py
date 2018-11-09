"""

Steady conduction with generation in an insulated wire

Carslaw, H. S., & J. C. Jaeger (1959). _Conduction of Heat in Solids_ 
(2nd ed.). Oxford University Press. §7.2.V, pp 191–192

∇ ⋅ (k0 ∇ T) + A = 0 in 0 < r < a

and 

∇ ⋅ (k1 ∇ T) = 0 in a < r < b

with k1 ∂T/∂r + h T = 0 on r = b.

"""

from typing import Optional

import numpy as np

from pygmsh import generate_mesh
from pygmsh.built_in import Geometry

from skfem.mesh import MeshTri
from skfem.models.poisson import laplace, unit_load

radii = [2., 3.]
joule_heating = 5.
heat_transfer_coefficient = 7.
thermal_conductivity = np.array([101.,  11.])


def make_mesh(a: float,         # radius of wire
              b: float,         # radius of insulation
              dx: Optional[float] = None) -> MeshTri:
    
    dx = a / 2 ** 3 if dx is None else dx

    origin = np.zeros(3)
    geom = Geometry()
    wire = geom.add_circle(origin, a, dx, make_surface=True)
    geom.add_physical_surface(wire.plane_surface, 'wire')
    insulation = geom.add_circle(origin, b, dx, holes=[wire.line_loop])
    geom.add_physical_surface(insulation.plane_surface, 'insulation')
    geom.add_physical_line(insulation.line_loop.lines, 'convection')

    points, cells, _, cell_data, __ = generate_mesh(geom)
    
    mesh = MeshTri(points[:, :2].T, cells['triangle'].T)
    mesh.regions = cell_data['triangle']['gmsh:physical']

    return mesh

mesh = make_mesh(*radii)
print(mesh)
print(mesh.regions)

ax = mesh.plot(mesh.regions)
ax.axis('off')
ax.get_figure().savefig('regions.png')
