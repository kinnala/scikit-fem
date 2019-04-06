from skfem import *
from skfem.models.poisson import laplace, unit_load

import numpy as np

from ex17 import mesh, basis, radii, joule_heating, thermal_conductivity


L = asm(laplace, basis)
f = asm(unit_load, basis)

insulation = np.unique(basis.element_dofs[:, mesh.subdomains['insulation']])
temperature = np.zeros(basis.N)
wire = basis.complement_dofs(insulation)
temperature[wire] = solve(*condense(thermal_conductivity['wire'] * L,
                                    joule_heating * f,
                                    D=insulation))

if __name__ == '__main__':

    from os.path import splitext
    from sys import argv

    T0 = {'skfem': basis.interpolator(temperature)(np.zeros((2, 1)))[0],
          'exact':
          joule_heating * radii[0]**2 / 4 / thermal_conductivity['wire']}
    print('Central temperature:', T0)

    mesh.plot(temperature[basis.nodal_dofs.flatten()], colorbar=True)
    mesh.savefig(splitext(argv[0])[0] + '_solution.png')
