import warnings

warnings.warn("DeprecationWarning: skfem.importers is removed "
              "in the next release. Use skfem.io instead.")

from ..io.meshio import *
