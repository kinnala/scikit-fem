"""Second-order forward-mode dual numbers for linearizing variational forms.

The seed groups are the scalar entries of the trial (A) and test (B)
fields at the quadrature points, e.g., the value and the gradient
components, so that the result of a form evaluation contains the
pointwise coefficients of the linearized form.

A :class:`Dual` holds a value array ``val`` together with

- ``ga``, the first derivatives with respect to the seed group A,
- ``gb``, the first derivatives with respect to the seed group B,
- ``hab``, the mixed second derivatives with respect to A and B.

The derivative axes are leading, i.e. if ``val`` has the shape ``S`` then
``ga`` has the shape ``(Ka,) + S``, ``gb`` has the shape ``(Kb,) + S``, and
``hab`` has the shape ``(Ka, Kb) + S``.  Derivative arrays may contain
singleton axes that broadcast against ``S``, and ``None`` stands for an
identically zero derivative.  If the two seed groups coincide (``gb is ga``)
then ``hab`` is the full Hessian.

"""
from string import ascii_lowercase

import numpy as np
from numpy import ndarray


FIELD_ATTRS = (
    'grad',
    'div',
    'curl',
    'hess',
    'grad3',
    'grad4',
    'grad5',
    'grad6',
)


def _const(x):
    """Convert a non-dual operand into a plain ndarray."""
    if isinstance(x, ndarray):
        # e.g. DiscreteField; drop the subclass
        return x.view(ndarray)
    return np.asarray(x)


def _val(x):
    return x.val if isinstance(x, Dual) else _const(x)



def _pad(d, nlead, ndim):
    """Insert singleton axes after the derivative axes of ``d``."""
    extra = ndim - (d.ndim - nlead)
    if extra <= 0:
        return d
    return d.reshape(d.shape[:nlead] + (1,) * extra + d.shape[nlead:])


def _full(d, nlead, shape):
    """Broadcast the value part of ``d`` to ``shape`` without copying."""
    d = _pad(d, nlead, len(shape))
    return np.broadcast_to(d, d.shape[:nlead]
                           + np.broadcast_shapes(d.shape[nlead:], shape))


def _scale(c, d, nlead, ndim):
    if d is None:
        return None
    return c * _pad(d, nlead, ndim)


def _add(nlead, ndim, *ds):
    out = None
    for d in ds:
        if d is None:
            continue
        d = _pad(d, nlead, ndim)
        out = d if out is None else out + d
    return out


def _outer(da, db, ndim):
    if da is None or db is None:
        return None
    da = _pad(da, 1, ndim)
    db = _pad(db, 1, ndim)
    return da[:, None] * db[None, :]


def _sym(*xs):
    """True if the seed groups A and B coincide for all operands."""
    return all(x.gb is x.ga for x in xs if isinstance(x, Dual))


class Dual:
    """Value and derivatives at the quadrature points.

    Trial and test functions passed to the forms are instances of this
    class and they carry the same extra attributes as
    :class:`~skfem.element.DiscreteField`, e.g., ``grad``.

    """

    def __init__(self, val, ga=None, gb=None, hab=None, **attrs):
        self.val = val
        self.ga = ga
        self.gb = gb
        self.hab = hab
        for attr in FIELD_ATTRS:
            setattr(self, attr, attrs.get(attr, None))

    @property
    def value(self):
        return Dual(self.val, self.ga, self.gb, self.hab)

    @property
    def shape(self):
        return self.val.shape

    @property
    def ndim(self):
        return self.val.ndim

    @property
    def dtype(self):
        return self.val.dtype

    def __len__(self):
        return len(self.val)

    def __repr__(self):
        return "<skfem Dual object of shape {}>".format(self.val.shape)

    def __array__(self, *args, **kwargs):
        raise TypeError("Cannot convert Dual to ndarray: the operation "
                        "would lose the derivatives.  Use NumPy functions "
                        "or skfem.autodiff.helpers instead.")

    def zeros(self):
        return np.zeros_like(self.val)

    # arithmetic

    def __add__(self, other):
        return add(self, other)

    def __radd__(self, other):
        return add(other, self)

    def __sub__(self, other):
        return subtract(self, other)

    def __rsub__(self, other):
        return subtract(other, self)

    def __mul__(self, other):
        return multiply(self, other)

    def __rmul__(self, other):
        return multiply(other, self)

    def __truediv__(self, other):
        return divide(self, other)

    def __rtruediv__(self, other):
        return divide(other, self)

    def __pow__(self, other):
        return power(self, other)

    def __rpow__(self, other):
        return power(other, self)

    def __neg__(self):
        return lincomb((-1., self))

    def __pos__(self):
        return self.value

    def __abs__(self):
        return absolute(self)

    def __lt__(self, other):
        return self.val < _val(other)

    def __le__(self, other):
        return self.val <= _val(other)

    def __gt__(self, other):
        return self.val > _val(other)

    def __ge__(self, other):
        return self.val >= _val(other)

    def __getitem__(self, idx):
        if not isinstance(idx, tuple):
            idx = (idx,)
        if any(isinstance(i, Dual) for i in idx):
            raise TypeError("Cannot index using Dual.")
        shape = self.val.shape

        def take(d, nlead):
            if d is None:
                return None
            return _full(d, nlead, shape)[(slice(None),) * nlead + idx]

        ga = take(self.ga, 1)
        return Dual(self.val[idx],
                    ga,
                    ga if self.gb is self.ga else take(self.gb, 1),
                    take(self.hab, 2))

    @property
    def T(self):
        return transpose(self)

    # numpy interoperability

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__' or len(kwargs) > 0:
            return NotImplemented
        if ufunc in _UFUNCS:
            return _UFUNCS[ufunc](*inputs)
        if ufunc in _NONDIFF_UFUNCS:
            return ufunc(*(_val(x) for x in inputs))
        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if func in _FUNCTIONS:
            return _FUNCTIONS[func](*args, **kwargs)
        return NotImplemented


# elementary operations


def lincomb(*terms):
    """Linear combination ``sum(c * x for c, x in terms)``.

    The coefficients ``c`` are constants; the operands ``x`` may be duals.

    """
    val = sum(c * _val(x) for c, x in terms)
    ndim = np.ndim(val)
    duals = [(c, x) for c, x in terms if isinstance(x, Dual)]
    ga = _add(1, ndim, *(_scale(c, x.ga, 1, ndim) for c, x in duals))
    if _sym(*(x for _, x in duals)):
        gb = ga
    else:
        gb = _add(1, ndim, *(_scale(c, x.gb, 1, ndim) for c, x in duals))
    hab = _add(2, ndim, *(_scale(c, x.hab, 2, ndim) for c, x in duals))
    return Dual(np.asarray(val), ga, gb, hab)


def unary(x, f0, f1, f2=None):
    """Chain rule for ``f(x)`` given ``f(x)``, ``f'(x)`` and ``f''(x)``."""
    ndim = np.ndim(f0)
    ga = _scale(f1, x.ga, 1, ndim)
    gb = ga if x.gb is x.ga else _scale(f1, x.gb, 1, ndim)
    hab = _scale(f1, x.hab, 2, ndim)
    if f2 is not None:
        hab = _add(2, ndim, hab, _scale(f2, _outer(x.ga, x.gb, ndim),
                                        2, ndim))
    return Dual(np.asarray(f0), ga, gb, hab)


def add(x, y):
    return lincomb((1., x), (1., y))


def subtract(x, y):
    return lincomb((1., x), (-1., y))


def multiply(x, y):
    if not isinstance(x, Dual):
        return lincomb((_const(x), y))
    if not isinstance(y, Dual):
        return lincomb((_const(y), x))
    val = x.val * y.val
    ndim = val.ndim
    ga = _add(1, ndim, _scale(y.val, x.ga, 1, ndim),
              _scale(x.val, y.ga, 1, ndim))
    if _sym(x, y):
        gb = ga
    else:
        gb = _add(1, ndim, _scale(y.val, x.gb, 1, ndim),
                  _scale(x.val, y.gb, 1, ndim))
    hab = _add(2, ndim,
               _scale(y.val, x.hab, 2, ndim),
               _scale(x.val, y.hab, 2, ndim),
               _outer(x.ga, y.gb, ndim),
               _outer(y.ga, x.gb, ndim))
    return Dual(val, ga, gb, hab)


def reciprocal(x):
    r = 1. / x.val
    return unary(x, r, -r ** 2, 2. * r ** 3)


def divide(x, y):
    if not isinstance(y, Dual):
        return lincomb((1. / _const(y), x))
    return multiply(x, reciprocal(y))


def power(x, y):
    if not isinstance(y, Dual):
        y = _const(y)
        if y.ndim == 0:
            if y == 0:
                return np.ones_like(x.val)
            if y == 1:
                return x.value
            if y == 2:
                return multiply(x, x)
        return unary(x,
                     x.val ** y,
                     y * x.val ** (y - 1),
                     y * (y - 1) * x.val ** (y - 2))
    if not isinstance(x, Dual):
        x = _const(x)
        f = x ** y.val
        return unary(y, f, f * np.log(x), f * np.log(x) ** 2)
    return exp(multiply(y, log(x)))


def _maxweight(x, y):
    xv, yv = _val(x), _val(y)
    wx = np.where(xv > yv, 1., np.where(xv == yv, .5, 0.))
    return wx, 1. - wx


def maximum(x, y):
    wx, wy = _maxweight(x, y)
    return lincomb((wx, x), (wy, y))


def minimum(x, y):
    wy, wx = _maxweight(x, y)
    return lincomb((wx, x), (wy, y))


def where(cond, x, y):
    cond = np.asarray(_val(cond), dtype=bool)
    return lincomb((cond * 1., x), ((~cond) * 1., y))


def sqrt(x):
    s = np.sqrt(x.val)
    return unary(x, s, .5 / s, -.25 / s ** 3)


def cbrt(x):
    c = np.cbrt(x.val)
    return unary(x, c, 1. / (3. * c ** 2), -2. / (9. * c ** 5))


def exp(x):
    e = np.exp(x.val)
    return unary(x, e, e, e)


def expm1(x):
    e = np.exp(x.val)
    return unary(x, np.expm1(x.val), e, e)


def log(x):
    r = 1. / x.val
    return unary(x, np.log(x.val), r, -r ** 2)


def log1p(x):
    r = 1. / (1. + x.val)
    return unary(x, np.log1p(x.val), r, -r ** 2)


def sin(x):
    s, c = np.sin(x.val), np.cos(x.val)
    return unary(x, s, c, -s)


def cos(x):
    s, c = np.sin(x.val), np.cos(x.val)
    return unary(x, c, -s, -c)


def tan(x):
    t = np.tan(x.val)
    return unary(x, t, 1. + t ** 2, 2. * t * (1. + t ** 2))


def sinh(x):
    s, c = np.sinh(x.val), np.cosh(x.val)
    return unary(x, s, c, s)


def cosh(x):
    s, c = np.sinh(x.val), np.cosh(x.val)
    return unary(x, c, s, c)


def tanh(x):
    t = np.tanh(x.val)
    return unary(x, t, 1. - t ** 2, -2. * t * (1. - t ** 2))


def arcsin(x):
    r = 1. / np.sqrt(1. - x.val ** 2)
    return unary(x, np.arcsin(x.val), r, x.val * r ** 3)


def arccos(x):
    r = 1. / np.sqrt(1. - x.val ** 2)
    return unary(x, np.arccos(x.val), -r, -x.val * r ** 3)


def arctan(x):
    r = 1. / (1. + x.val ** 2)
    return unary(x, np.arctan(x.val), r, -2. * x.val * r ** 2)


def absolute(x):
    return lincomb((np.sign(x.val), x))


def square(x):
    return multiply(x, x)


def negative(x):
    return lincomb((-1., x))


def positive(x):
    return x.value


# multilinear operations


def _explicit(subscripts, nops):
    """Make einsum subscripts explicit, i.e. add '->' if missing."""
    subscripts = subscripts.replace(' ', '')
    if '->' in subscripts:
        return subscripts
    inputs = subscripts.split(',')
    letters = ''.join(inputs).replace('.', '')
    out = ''.join(sorted(c for c in set(letters) if letters.count(c) == 1))
    if '...' in subscripts:
        out = '...' + out
    return subscripts + '->' + out


def einsum(subscripts, *operands, **kwargs):
    """Dual-aware :func:`numpy.einsum` which is multilinear in operands."""
    subscripts = _explicit(subscripts, len(operands))
    ins, out = subscripts.split('->')
    ins = ins.split(',')
    vals = [_val(x) for x in operands]
    val = np.einsum(subscripts, *vals, **kwargs)
    # letters for the derivative axes
    a, b = [c for c in ascii_lowercase if c not in subscripts][:2]

    def full(x, attr, nlead):
        return _full(getattr(x, attr), nlead, x.val.shape)

    def deriv(prefixes, outprefix, ops):
        spec = (','.join(p + s for p, s in zip(prefixes, ins))
                + '->' + outprefix + out)
        return np.einsum(spec, *ops, **kwargs)

    n = len(operands)
    duals = [k for k in range(n) if isinstance(operands[k], Dual)]

    def first(attr, letter):
        res = None
        for k in duals:
            if getattr(operands[k], attr) is None:
                continue
            ops = list(vals)
            ops[k] = full(operands[k], attr, 1)
            pre = [''] * n
            pre[k] = letter
            res = _add(1, val.ndim, res, deriv(pre, letter, ops))
        return res

    ga = first('ga', a)
    gb = ga if _sym(*operands) else first('gb', b)
    hab = None
    for k in duals:
        if operands[k].hab is not None:
            ops = list(vals)
            ops[k] = full(operands[k], 'hab', 2)
            pre = [''] * n
            pre[k] = a + b
            hab = _add(2, val.ndim, hab, deriv(pre, a + b, ops))
        for m in duals:
            if (m == k
                    or operands[k].ga is None
                    or operands[m].gb is None):
                continue
            ops = list(vals)
            ops[k] = full(operands[k], 'ga', 1)
            ops[m] = full(operands[m], 'gb', 1)
            pre = [''] * n
            pre[k] = a
            pre[m] = b
            hab = _add(2, val.ndim, hab, deriv(pre, a + b, ops))
    return Dual(val, ga, gb, hab)


def _linear(x, fun, nlead_fun):
    """Apply a linear map defined on the value part of ``x``."""
    shape = x.val.shape

    def apply(d, nlead):
        if d is None:
            return None
        return nlead_fun(_full(d, nlead, shape), nlead)

    ga = apply(x.ga, 1)
    return Dual(fun(x.val),
                ga,
                ga if x.gb is x.ga else apply(x.gb, 1),
                apply(x.hab, 2))


def _sum(x, axis=None, **kwargs):
    if not isinstance(x, Dual):
        return np.sum(x, axis=axis, **kwargs)
    ndim = x.val.ndim
    if axis is None:
        axis = tuple(range(ndim))
    elif not isinstance(axis, tuple):
        axis = (axis,)
    axis = tuple(ax % ndim for ax in axis)
    return _linear(x,
                   lambda v: np.sum(v, axis=axis),
                   lambda d, n: np.sum(d, axis=tuple(ax + n for ax in axis)))


def transpose(x, axes=None):
    ndim = x.val.ndim
    if axes is None:
        axes = tuple(range(ndim))[::-1]
    return _linear(x,
                   lambda v: np.transpose(v, axes),
                   lambda d, n: np.transpose(
                       d, tuple(range(n)) + tuple(ax + n for ax in axes)))


def trace(x, offset=0, axis1=0, axis2=1):
    return _linear(x,
                   lambda v: np.trace(v, offset, axis1, axis2),
                   lambda d, n: np.trace(d, offset, axis1 + n, axis2 + n))


def stack(arrays, axis=0):
    """Dual-aware :func:`numpy.stack`."""
    arrays = list(arrays)
    vals = np.broadcast_arrays(*(_val(x) for x in arrays))
    val = np.stack(vals, axis=axis)
    duals = [x for x in arrays if isinstance(x, Dual)]
    if len(duals) == 0:
        return val
    shape = vals[0].shape
    axis = axis % val.ndim

    def combine(attr, nlead):
        ds = [getattr(x, attr) for x in duals]
        if all(d is None for d in ds):
            return None
        lead = next(d for d in ds if d is not None).shape[:nlead]
        zero = np.zeros(lead + shape)
        out = []
        for x in arrays:
            d = getattr(x, attr) if isinstance(x, Dual) else None
            out.append(zero if d is None else
                       np.broadcast_to(_pad(d, nlead, len(shape)),
                                       lead + shape))
        return np.stack(out, axis=axis + nlead)

    ga = combine('ga', 1)
    return Dual(val,
                ga,
                ga if _sym(*duals) else combine('gb', 1),
                combine('hab', 2))


def zeros_like(x, *args, **kwargs):
    return np.zeros_like(_val(x), *args, **kwargs)


def ones_like(x, *args, **kwargs):
    return np.ones_like(_val(x), *args, **kwargs)


def _shape(x):
    return np.shape(_val(x))


def _ndim(x):
    return np.ndim(_val(x))


_UFUNCS = {
    np.add: add,
    np.subtract: subtract,
    np.multiply: multiply,
    np.divide: divide,
    np.true_divide: divide,
    np.power: power,
    np.float_power: power,
    np.maximum: maximum,
    np.minimum: minimum,
    np.fmax: maximum,
    np.fmin: minimum,
    np.negative: negative,
    np.positive: positive,
    np.absolute: absolute,
    np.square: square,
    np.reciprocal: reciprocal,
    np.sqrt: sqrt,
    np.cbrt: cbrt,
    np.exp: exp,
    np.expm1: expm1,
    np.log: log,
    np.log1p: log1p,
    np.sin: sin,
    np.cos: cos,
    np.tan: tan,
    np.sinh: sinh,
    np.cosh: cosh,
    np.tanh: tanh,
    np.arcsin: arcsin,
    np.arccos: arccos,
    np.arctan: arctan,
}

# piecewise constant functions and predicates: evaluated on values only
_NONDIFF_UFUNCS = {
    np.sign,
    np.floor,
    np.ceil,
    np.rint,
    np.greater,
    np.greater_equal,
    np.less,
    np.less_equal,
    np.equal,
    np.not_equal,
    np.isnan,
    np.isinf,
    np.isfinite,
}

_FUNCTIONS = {
    np.einsum: einsum,
    np.sum: _sum,
    np.transpose: transpose,
    np.trace: trace,
    np.stack: stack,
    np.where: where,
    np.zeros_like: zeros_like,
    np.ones_like: ones_like,
    np.shape: _shape,
    np.ndim: _ndim,
}
