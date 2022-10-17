"""Support for wildcard import."""

from skfem.mesh import *  # noqa
from skfem.assembly import *  # noqa
from skfem.mapping import *  # noqa
from skfem.element import *  # noqa
from skfem.utils import *  # noqa

from skfem.assembly import __all__ as all_assembly
from skfem.mesh import __all__ as all_mesh
from skfem.element import __all__ as all_element

from .__about__ import __version__


__all__ = all_mesh + all_assembly + all_element + [  # noqa
    'MappingAffine',
    'MappingIsoparametric',
    'MappingMortar',  # TODO remove due to deprecation
    'adaptive_theta',
    'build_pc_ilu',
    'build_pc_diag',
    'condense',
    'enforce',
    'penalize',
    'project',  # TODO remove due to deprecation
    'projection',  # TODO remove due to deprecation
    'solve',
    'bmat',
    'solver_direct_scipy',
    'solver_eigen_scipy',
    'solver_eigen_scipy_sym',
    'solver_iter_pcg',
    'solver_iter_krylov',
    'solver_iter_cg',
    '__version__',
]
