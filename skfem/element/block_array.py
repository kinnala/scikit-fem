"""Arrays with most leading components equal to zero.

:class:`~skfem.element.ElementVector` and
:class:`~skfem.element.ElementComposite` produce basis functions whose
quadrature point values are mostly zero. Storing such fields densely
causes the forms spend most of their time multiplying zeros.

:class:`BlockArray` stores only the nonzero components and implements
the operations used in forms without touching the zeros.  Any other
operation converts the array to a dense ``ndarray``.

"""
import re
from itertools import product
from typing import Dict, Tuple

import numpy as np
from numpy import ndarray
from numpy.lib.mixins import NDArrayOperatorsMixin


def _dense(x):
    return np.asarray(x) if isinstance(x, BlockArray) else x


def _shape(x):
    return x.shape if isinstance(x, BlockArray) else np.shape(x)


def _dtype(x):
    return x.dtype if isinstance(x, BlockArray) else np.asarray(x).dtype


class BlockArray(NDArrayOperatorsMixin):
    """An array of shape ``shape`` with ``r`` sparse leading axes.

    The array is zero except for the entries listed in ``blocks``, a
    dictionary mapping tuples of length ``r`` (indices to the leading axes)
    to dense arrays of shape ``shape[r:]``.

    """

    _einsum_subscript = re.compile(r'^([a-zA-Z]*)\.\.\.$')

    def __init__(self,
                 shape: Tuple[int, ...],
                 r: int,
                 blocks: Dict[Tuple[int, ...], ndarray],
                 dtype=np.float64):
        self.shape = tuple(shape)
        self.r = r
        self.blocks = blocks
        self.dtype = np.dtype(dtype)

    @classmethod
    def wrap(cls, shape, r, blocks, dtype):
        """Avoid wrapping a single dense block."""
        if r == 0 and () in blocks:
            return blocks[()]
        return cls(shape, r, blocks, dtype)

    @classmethod
    def zeros(cls, shape, dtype=np.float64):
        return cls(shape, 0, {}, dtype)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return int(np.prod(self.shape))

    def __array__(self, dtype=None, copy=None):
        out = np.zeros(self.shape, dtype=self.dtype if dtype is None
                       else dtype)
        for key, block in self.blocks.items():
            out[key] = block
        return out

    def __getattr__(self, name):
        # fall back to the dense array for, e.g., '.T' or '.sum()'
        if name.startswith('__') or name in ('shape', 'r', 'blocks', 'dtype'):
            raise AttributeError(name)
        return getattr(np.asarray(self), name)

    def __repr__(self):
        return "<skfem BlockArray shape={} nonzero blocks={}>".format(
            self.shape,
            sorted(self.blocks)
        )

    def __len__(self):
        return self.shape[0]

    def __iter__(self):
        for i in range(self.shape[0]):
            yield self[i]

    def __getitem__(self, key):
        key = key if isinstance(key, tuple) else (key,)
        m = 0
        while (m < len(key) and m < self.r
               and isinstance(key[m], (int, np.integer))):
            m += 1
        if m == 0:
            return np.asarray(self)[key]
        head = []
        for i, k in enumerate(key[:m]):
            if not -self.shape[i] <= k < self.shape[i]:
                raise IndexError("index {} is out of bounds for axis {} "
                                 "with size {}".format(k, i, self.shape[i]))
            head.append(int(k) % self.shape[i])
        rest = key[m:]
        if m < self.r:
            if any(k is not Ellipsis and k != slice(None) for k in rest):
                return np.asarray(self)[key]
            blocks = {k[m:]: b
                      for k, b in self.blocks.items()
                      if k[:m] == tuple(head)}
            return type(self).wrap(self.shape[m:],
                                   self.r - m,
                                   blocks,
                                   self.dtype)
        block = self.blocks.get(tuple(head))
        if block is None:
            if len(rest) == 0:
                return BlockArray.zeros(self.shape[m:], self.dtype)
            return np.zeros(self.shape[m:], dtype=self.dtype)[rest]
        return block[rest] if len(rest) > 0 else block

    def promote(self, r: int) -> 'BlockArray':
        """Split the blocks so that there are ``r`` sparse axes."""
        if self.r >= r:
            return self
        blocks = {}
        for key, block in self.blocks.items():
            for ix in np.ndindex(*block.shape[:r - self.r]):
                blocks[key + ix] = block[ix]
        return BlockArray(self.shape, r, blocks, self.dtype)

    def _aligned(self, other, key):
        """Part of a dense ``other`` that multiplies the block ``key``."""
        other = np.asarray(other)
        if other.ndim <= self.ndim - self.r:
            return other
        other = other.reshape((1,) * (self.ndim - other.ndim) + other.shape)
        return other[tuple(k if other.shape[i] > 1 else 0
                           for i, k in enumerate(key))]

    # in-place operations create new objects
    def __iadd__(self, other):
        return self + other

    def __isub__(self, other):
        return self - other

    def __imul__(self, other):
        return self * other

    def __itruediv__(self, other):
        return self / other

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != '__call__' or kwargs:
            return getattr(ufunc, method)(*map(_dense, inputs), **kwargs)
        out = self._ufunc(ufunc, *inputs)
        if out is NotImplemented:
            return ufunc(*map(_dense, inputs))
        return out

    def _ufunc(self, ufunc, *inputs):
        sparse = [x for x in inputs if isinstance(x, BlockArray)]
        a = sparse[0]
        dtype = np.result_type(*map(_dtype, inputs))

        if ufunc in (np.negative, np.positive):
            return BlockArray(a.shape,
                              a.r,
                              {k: ufunc(b) for k, b in a.blocks.items()},
                              a.dtype)

        if len(inputs) != 2:
            return NotImplemented
        x, y = inputs
        try:
            shape = np.broadcast_shapes(_shape(x), _shape(y))
        except ValueError:
            return NotImplemented

        if ufunc is np.multiply:
            if any(len(z.blocks) == 0 for z in sparse):
                return BlockArray.zeros(shape, dtype)
            if len(sparse) == 2:
                if x.shape != y.shape:
                    return NotImplemented
                r = max(x.r, y.r)
                x, y = x.promote(r), y.promote(r)
                keys = x.blocks.keys() & y.blocks.keys()
                return BlockArray.wrap(shape,
                                       r,
                                       {k: x.blocks[k] * y.blocks[k]
                                        for k in keys},
                                       dtype)
            if shape != a.shape:
                return NotImplemented
            other = y if x is a else x
            return BlockArray.wrap(shape,
                                   a.r,
                                   {k: b * a._aligned(other, k)
                                    for k, b in a.blocks.items()},
                                   dtype)

        if ufunc is np.true_divide and len(sparse) == 1 and x is a:
            if shape != a.shape:
                return NotImplemented
            return BlockArray(shape,
                              a.r,
                              {k: b / a._aligned(y, k)
                               for k, b in a.blocks.items()},
                              dtype)

        if ufunc in (np.add, np.subtract):
            if len(sparse) == 2 and x.shape == y.shape:
                r = max(x.r, y.r)
                x, y = x.promote(r), y.promote(r)
                blocks = dict(x.blocks)
                for k, b in y.blocks.items():
                    if k in blocks:
                        blocks[k] = ufunc(blocks[k], b)
                    else:
                        blocks[k] = b if ufunc is np.add else -b
                return BlockArray.wrap(shape, r, blocks, dtype)
            if len(sparse) == 1 and len(a.blocks) == 0:
                # adding zero, e.g., the initial value of sum()
                other = y if x is a else x
                if np.ndim(other) == 0 and other == 0:
                    return a
                if _shape(other) == shape:
                    if ufunc is np.subtract and x is a:
                        return -other
                    return other

        return NotImplemented

    def __array_function__(self, func, types, args, kwargs):
        if (func is np.einsum
                and len(kwargs) == 0
                and isinstance(args[0], str)):
            out = _einsum(args[0], *args[1:])
            if out is not NotImplemented:
                return out
        return func(*[_dense(arg) for arg in args],
                    **{k: _dense(v) for k, v in kwargs.items()})


def _einsum(subscripts, *operands):
    """Evaluate einsum for subscripts such as 'ij...,j...->i...'.

    Loops over the combinations of nonzero blocks and evaluates a smaller
    einsum for each consistent combination.

    """
    subscripts = subscripts.replace(' ', '')
    if '->' in subscripts:
        lhs, rhs = subscripts.split('->')
        inputs = lhs.split(',')
    else:
        inputs = subscripts.split(',')
        letters = ''.join(i.replace('...', '') for i in inputs)
        if any(letters.count(c) == 1 for c in letters):
            # implicit output would be '...ij'
            return NotImplemented
        rhs = '...'
    if len(inputs) != len(operands):
        return NotImplemented
    matches = [BlockArray._einsum_subscript.match(s) for s in inputs + [rhs]]
    if not all(matches):
        return NotImplemented
    labels = [m.group(1) for m in matches[:-1]]
    olabels = matches[-1].group(1)
    for lab, op in zip(labels, operands):
        if isinstance(op, BlockArray) and len(lab) < op.r:
            return NotImplemented

    # output shape
    sizes = {}
    trailing = []
    for lab, op in zip(labels, operands):
        sizes.update(zip(lab, _shape(op)[:len(lab)]))
        trailing.append(_shape(op)[len(lab):])
    oshape = (tuple(sizes[c] for c in olabels)
              + np.broadcast_shapes(*trailing))
    dtype = np.result_type(*map(_dtype, operands))

    sparse = [i for i, op in enumerate(operands)
              if isinstance(op, BlockArray)]
    bound = set(c for i in sparse for c in labels[i][:operands[i].r])
    # the output is sparse up to its last bound label; any free labels
    # before that are split into separate blocks
    r = max([j + 1 for j, c in enumerate(olabels) if c in bound], default=0)
    nsplit = sum(1 for c in olabels[:r] if c not in bound)
    osub = ''.join(c for c in olabels if c not in bound) + '...'
    dense = [None if i in sparse else np.asarray(op)
             for i, op in enumerate(operands)]

    blocks: Dict[Tuple[int, ...], ndarray] = {}
    for combo in product(*(operands[i].blocks.items() for i in sparse)):
        bind: Dict[str, int] = {}
        consistent = True
        for i, (key, _) in zip(sparse, combo):
            for c, k in zip(labels[i], key):
                if bind.setdefault(c, k) != k:
                    consistent = False
        if not consistent:
            continue
        subs = []
        arrays = []
        combo_blocks = {i: block for i, (_, block) in zip(sparse, combo)}
        for i, lab in enumerate(labels):
            if i in combo_blocks:
                lab = lab[operands[i].r:]
                arr = combo_blocks[i]
            else:
                arr = dense[i]
            ix = tuple(bind[c] if c in bind else slice(None) for c in lab)
            subs.append(''.join(c for c in lab if c not in bind) + '...')
            arrays.append(arr[ix] if len(ix) > 0 else arr)
        val = np.einsum(','.join(subs) + '->' + osub, *arrays)
        for fix in np.ndindex(*val.shape[:nsplit]):
            it = iter(fix)
            key = tuple(bind[c] if c in bound else next(it)
                        for c in olabels[:r])
            if key in blocks:
                blocks[key] = blocks[key] + val[fix]
            else:
                blocks[key] = val[fix]

    return BlockArray.wrap(oshape, r, blocks, dtype)


class BlockDiscreteField(BlockArray):
    """A :class:`~skfem.element.DiscreteField` with sparse components."""

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

    def __init__(self, value, *derivatives):
        super().__init__(value.shape, value.r, value.blocks, value.dtype)
        for i, attr in enumerate(self._extra_attrs):
            setattr(self,
                    attr,
                    derivatives[i] if i < len(derivatives) else None)

    def get(self, n):
        if n == 0:
            return BlockArray(self.shape, self.r, self.blocks, self.dtype)
        return getattr(self, self._extra_attrs[n - 1])

    @property
    def astuple(self):
        return tuple(self.get(i) for i in range(len(self._extra_attrs) + 1))

    @property
    def value(self):
        return np.asarray(self)

    def zeros(self):
        return BlockDiscreteField(*(None if c is None
                                    else BlockArray.zeros(c.shape, c.dtype)
                                    for c in self.astuple))

    def __repr__(self):
        rep = "<skfem BlockDiscreteField object>"
        rep += "\n  Shape: {}".format(self.shape)
        rep += "\n  Nonzero blocks: {}".format(sorted(self.blocks))
        attrs = ', '.join([attr
                           for attr in self._extra_attrs
                           if getattr(self, attr) is not None])
        if len(attrs) > 0:
            rep += "\n  Attributes: {}".format(attrs)
        return rep
