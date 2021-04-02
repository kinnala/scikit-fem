"""Support for wildcard import."""

from skfem.mesh import *  # noqa
from skfem.assembly import *  # noqa
from skfem.mapping import *  # noqa
from skfem.element import *  # noqa
from skfem.utils import *  # noqa

from skfem.assembly import __all__ as all_assembly
from skfem.mesh import __all__ as all_mesh
from skfem.element import __all__ as all_element


__all__ = all_mesh + all_assembly + all_element + [  # noqa
    'MappingAffine',
    'MappingIsoparametric',
    'MappingMortar',
    'adaptive_theta',
    'build_pc_ilu',
    'build_pc_diag',
    'condense',
    'enforce',
    'project',
    'projection',
    'solve',
    'solver_direct_scipy',
    'solver_eigen_scipy',
    'solver_eigen_scipy_sym',
    'solver_iter_pcg',
    'solver_iter_krylov',
]
