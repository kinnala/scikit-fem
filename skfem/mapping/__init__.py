"""Mappings define relationships between reference and global elements.

:class:`~skfem.mesh.Mesh` provides default mappings for each mesh type,
so normally the user is not required to initialize these classes.

"""

from .mapping import Mapping  # noqa
from .mapping_affine import MappingAffine  # noqa
from .mapping_isoparametric import MappingIsoparametric  # noqa
from .mapping_mortar import MappingMortar  # noqa
