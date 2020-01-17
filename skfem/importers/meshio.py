import warnings

warnings.warn("DeprecationWarning: skfem.importers was renamed to "
              "skfem.io and is removed in the next major release.")

from ..io.meshio import *
