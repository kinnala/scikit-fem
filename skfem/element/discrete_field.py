from typing import NamedTuple, Optional

import numpy as np
from numpy import ndarray


class DiscreteField(NamedTuple):
    """A function defined at the global quadrature points."""

    value: ndarray
    grad: Optional[ndarray] = None
    div: Optional[ndarray] = None
    curl: Optional[ndarray] = None
    hess: Optional[ndarray] = None
    grad3: Optional[ndarray] = None
    grad4: Optional[ndarray] = None
    grad5: Optional[ndarray] = None
    grad6: Optional[ndarray] = None

    def __array__(self):
        return self.value

    def __add__(self, other):
        if isinstance(other, DiscreteField):
            return self.value + other.value
        return self.value + other

    def __sub__(self, other):
        if isinstance(other, DiscreteField):
            return self.value - other.value
        return self.value - other

    def __mul__(self, other):
        if isinstance(other, DiscreteField):
            return self.value * other.value
        return self.value * other

    def __truediv__(self, other):
        if isinstance(other, DiscreteField):
            return self.value / other.value
        return self.value / other

    def __pow__(self, other):
        if isinstance(other, DiscreteField):
            return self.value ** other.value
        return self.value ** other

    def __neg__(self):
        return -self.value

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        return other - self.value

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        return other / self.value

    def __rpow__(self, other):
        return other ** self.value

    def _split(self):
        """Split all components based on their first dimension."""
        return [DiscreteField(*[f[i] for f in self if f is not None])
                for i in range(self.value.shape[0])]

    def zeros_like(self) -> 'DiscreteField':
        """Return zero :class:`~skfem.element.DiscreteField` with same size."""

        def zero_or_none(x):
            if x is None:
                return None
            return np.zeros_like(x)

        return DiscreteField(*[zero_or_none(field) for field in self])
