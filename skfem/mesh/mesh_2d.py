from dataclasses import dataclass

import numpy as np
from numpy import ndarray

from .mesh import Mesh


@dataclass(repr=False)
class Mesh2D(Mesh):

    def param(self) -> float:
        return np.max(
            np.linalg.norm(np.diff(self.p[:, self.facets], axis=1), axis=0)
        )

    @staticmethod
    def strip_extra_coordinates(p: ndarray) -> ndarray:
        """For meshio which appends :math:`z = 0` to 2D meshes."""
        return p[:, :2]

    def _repr_svg_(self) -> str:
        from skfem.visuals.svg import draw
        return draw(self, nrefs=0, boundaries_only=True).svg
