from warnings import warn

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
                value=None,
                grad=None,
                div=None,
                curl=None,
                hess=None,
                grad3=None,
                grad4=None,
                grad5=None,
                grad6=None):
        if value is None:
            value = np.array([0])  # TODO unused?
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

    def __getitem__(self, key):
        return np.array(self)[key]

    def __array_wrap__(self, out_arr, context=None):
        # invalidate attributes after ufuncs
        return np.array(out_arr)

    def get(self, n):
        if n == 0:
            return np.array(self)
        return getattr(self, self._extra_attrs[n - 1])

    @property
    def astuple(self):
        return tuple(self.get(i) for i in range(len(self._extra_attrs) + 1))

    @property
    def value(self):
        # for backwards-compatibility
        warn("Writing 'u.value' is unnecessary "
             "and can be replaced by 'u'.", DeprecationWarning)
        return np.array(self)

    def zeros(self):
        return DiscreteField(*tuple(None if c is None else np.zeros_like(c)
                                    for c in self.astuple))

    def __repr__(self):
        rep = ""
        rep += "<skfem DiscreteField object>"
        rep += ("\n  Quadrature points per element: {}"
                .format(self.shape[-1]))
        rep += "\n  Number of elements: {}".format(self.shape[-2])
        rep += "\n  Order: {}".format(len(self.shape) - 2)
        attrs = ', '.join([attr
                           for attr in self._extra_attrs
                           if getattr(self, attr) is not None])
        if len(attrs) > 0:
            rep += "\n  Attributes: {}".format(attrs)
        return rep

    def __reduce__(self):
        # for pickling
        pickled_state = super(DiscreteField, self).__reduce__()
        new_state = pickled_state[2] + self.astuple[1:]
        return (pickled_state[0], pickled_state[1], new_state)

    def __setstate__(self, state):
        # for pickling
        nattrs = len(self._extra_attrs)
        for i in range(nattrs):
            setattr(self, self._extra_attrs[i], state[-nattrs + i])
        super(DiscreteField, self).__setstate__(state[0:-nattrs])
