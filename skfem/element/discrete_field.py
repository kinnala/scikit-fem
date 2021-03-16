from typing import Optional

import numpy as np
from numpy import ndarray
from dataclasses import dataclass


@dataclass
class DiscreteField:
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

    def __post_init__(self):
        self.as_tuple = (self.value, self.grad, self.div, self.curl,
                         self.hess, self.grad3, self.grad4,
                         self.grad5, self.grad6)
        all_data = tuple(f.tobytes() if f is not None else None for f in self)
        self._hash = hash(all_data)

    def __getitem__(self, item):
        return self.as_tuple[item]

    def __len__(self):
        return len(self.as_tuple)

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if not isinstance(other, DiscreteField):
            return False
        for i in range(len(self.as_tuple)):
            if self[i] is None and other[i] is None:
                continue
            elif self[i] is None or other[i] is None:
                return False
            elif (self[i] == other[i]).all():
                continue
            else:
                return False
        return True

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

        return DiscreteField(*[zero_or_none(field) for field in self.as_tuple])
