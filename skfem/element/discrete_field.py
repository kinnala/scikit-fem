from typing import NamedTuple, Optional

from numpy import ndarray


class DiscreteField(NamedTuple):
    """A function defined at the global quadrature points."""

    f: Optional[ndarray] = None
    df: Optional[ndarray] = None
    ddf: Optional[ndarray] = None

    def __array__(self):
        return self.f

    def __mul__(self, other):
        if isinstance(other, DiscreteField):
            return self.f * other.f
        return self.f * other

    def _split(self):
        """Split all components based on their first dimension."""
        return [DiscreteField(*[f[i] for f in self if f is not None])
                for i in range(self.f.shape[0])]

    def zeros_like(self):
        return DiscreteField(*[0. * field.copy()
                               for field in self
                               if field is not None])

    __rmul__ = __mul__
