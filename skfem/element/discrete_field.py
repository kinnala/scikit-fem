import numpy as np
from numpy import ndarray


class DiscreteField(ndarray):

    _extra_attrs = (
        'grad',
        'div',
        'curl',
        'hess',
        'grad3',
        'grad4',
        'grad5',
        'grad6',
    )

    def __new__(cls,
                value=np.array([0]),
                grad=None,
                div=None,
                curl=None,
                hess=None,
                grad3=None,
                grad4=None,
                grad5=None,
                grad6=None):
        obj = np.asarray(value).view(cls)
        obj.grad = grad
        obj.div = div
        obj.curl = curl
        obj.hess = hess
        obj.grad3 = grad3
        obj.grad4 = grad4
        obj.grad5 = grad5
        obj.grad6 = grad6
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return
        self.grad = getattr(obj, 'grad', None)
        self.div = getattr(obj, 'div', None)
        self.curl = getattr(obj, 'curl', None)
        self.hess = getattr(obj, 'hess', None)
        self.grad3 = getattr(obj, 'grad3', None)
        self.grad4 = getattr(obj, 'grad4', None)
        self.grad5 = getattr(obj, 'grad5', None)
        self.grad6 = getattr(obj, 'grad6', None)

    @property
    def astuple(self):
        return tuple(self.get(i) for i in range(len(self._extra_attrs) + 1))

    @property
    def value(self):
        return self

    def get(self, n):
        if n == 0:
            return np.array(self)
        return getattr(self, self._extra_attrs[n - 1])

    def is_zero(self):
        return self.shape == (1,)

    def __reduce__(self):
        pickled_state = super(DiscreteField, self).__reduce__()
        new_state = pickled_state[2] + self.astuple[1:]
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        nattrs = len(self._extra_attrs)
        for i in range(nattrs):
            setattr(self, self._extra_attrs[i], state[-nattrs + i])
        super(DiscreteField, self).__setstate__(state[0:-nattrs])
