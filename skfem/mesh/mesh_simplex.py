from dataclasses import replace

import numpy as np


class MeshSimplex:
    """Mixin for simplical meshes."""

    def orientation(self):
        """Return the sign of the Jacobian determinant for each element."""
        mapping = self._mapping()
        return (np.sign(mapping.detDF(np.zeros(self.p.shape[0])[:, None]))
                .flatten()
                .astype(np.int32))

    def oriented(self):
        """Return a oriented mesh with positive Jacobian determinant.

        For triangular meshes this corresponds to CCW orientation.

        """
        flip = np.nonzero(self.orientation() == -1)[0].astype(np.int32)
        t = self.t.copy()
        t0 = t[0, flip]
        t1 = t[1, flip]
        t[0, flip] = t1
        t[1, flip] = t0

        return replace(
            self,
            t=t,
            sort_t=False,
        )
