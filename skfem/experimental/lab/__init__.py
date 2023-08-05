from skfem import *
from skfem.experimental.autodiff import NonlinearForm
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from itertools import chain


@dataclass
class Tensor:
    expr: str
    order: int = 0
    args: Optional[List] = None
    dyadic: bool = False
    trial: Optional[List] = None
    test: Optional[List] = None

    def __iter__(self):
        if self.args is not None:
            for v in chain.from_iterable(self.args):
                yield v
        yield self

    def __repr__(self):
        if self.args is not None:
            if self.dyadic:
                assert len(self.args) == 2
                return ('('
                        + repr(self.args[0])
                        + self.expr
                        + repr(self.args[1])
                        + ')')
            return (self.expr
                    + '('
                    + ','.join(map(lambda x: repr(x), self.args))
                    + ')')
        return self.expr

    def _legacy_repr(self):
        out = str(self)
        for p in self._params():
            out = out.replace('grad({})'.format(p), 'GRAD')
            out = out.replace('div({})'.format(p), 'DIV')
            out = out.replace(p, p + "[0]")
            out = out.replace('GRAD', 'grad({})'.format(p))
            out = out.replace('DIV', 'div({})'.format(p))
        return out

    def _trial_params(self):
        for t in self:
            if t.trial is not None:
                return t.trial
        return []

    def _test_params(self):
        for t in self:
            if t.test is not None:
                return t.test
        return []

    def _params(self) -> str:
        return [] + self._trial_params() + self._test_params()

    @property
    def T(self):
        assert self.order == 2
        return Tensor('transpose', self.order, [self])

    def __add__(self, other):
        assert self.order == other.order
        return Tensor('+', self.order, [self, other], True)

    def __sub__(self, other):
        assert self.order == other.order
        return Tensor('-', self.order, [self, other], True)

    def __mul__(self, other):
        if isinstance(other, float) or isinstance(other, int):
            other = Tensor(repr(other), order=0)
        assert self.order == 0 or other.order == 0
        return Tensor('*', max(self.order, other.order), [self, other], True)

    def __rmul__(self, other):
        return self * other

    def __matmul__(self, other):
        return mul(self, other)

    def __getitem__(self, key):
        assert self.order > 0
        assert isinstance(key, int)
        assert key >= 0
        return Tensor('_', self.order - 1, [self, Tensor(repr(key), order=0)])

    def assemble(self, *args, **kwargs):
        assert self.order == 0
        if 'x' in kwargs:
            x = kwargs['x']
            del kwargs['x']
            if len(self._test_params()) == 0:
                return NonlinearForm(hessian=True)(
                    self._as_legacy_fun()
                ).assemble(*args, x=x, **kwargs)
            return (NonlinearForm(self._as_legacy_fun())
                    .assemble(*args, x=x, **kwargs))
        if len(self._trial_params()) == 0:
            return LinearForm(self._as_fun()).assemble(*args, **kwargs)
        return BilinearForm(self._as_fun()).assemble(*args, **kwargs)

    def _as_fun(self):
        from skfem.helpers import (dot,
                                   ddot,
                                   grad,
                                   div,
                                   transpose,
                                   trace,
                                   mul,
                                   eye)
        scope = {}
        params = ','.join(self._params() + ['w']) + ':'
        exec('fun=lambda ' + params + str(self),
             {'dot': dot,
              'ddot': ddot,
              'grad': grad,
              'div': div,
              'transpose': transpose,
              'trace': trace,
              'sin': np.sin,
              'cos': np.cos,
              'tan': np.tan,
              'mul': mul,
              'eye': eye,
              '_': lambda a, i: a[i]},
             scope)
        return scope['fun']

    def _as_legacy_fun(self):
        from skfem.experimental.autodiff.helpers import (dot,
                                                         ddot,
                                                         grad,
                                                         div,
                                                         transpose,
                                                         trace,
                                                         mul,
                                                         eye)
        import autograd.numpy as anp
        scope = {}
        params = ','.join(self._params() + ['w']) + ':'
        exec('fun=lambda ' + params + self._legacy_repr(),
             {'dot': dot,
              'ddot': ddot,
              'grad': grad,
              'div': div,
              'transpose': transpose,
              'trace': trace,
              'sin': anp.sin,
              'cos': anp.cos,
              'tan': anp.tan,
              'mul': mul,
              'eye': eye,
              '_': lambda a, i: a[i]},
             scope)
        return scope['fun']

    def coo_data(self, *args, **kwargs):
        NotImplementedError


def _find_order(elem):
    if isinstance(elem, ElementVector):
        return 1
    return 0


def _trial_functions(elem, expr='u'):
    if isinstance(elem, ElementComposite):
        elems = elem.elems
    else:
        elems = [elem]
    trials = ['{}{}'.format(expr, i + 1)
              for i, e in enumerate(elems)]
    return [Tensor(trials[i], order=_find_order(e), trial=trials)
            for i, e in enumerate(elems)]


def _test_functions(elem, expr='v'):
    if isinstance(elem, ElementComposite):
        elems = elem.elems
    else:
        elems = [elem]
    tests = ['{}{}'.format(expr, i + 1)
              for i, e in enumerate(elems)]
    return [Tensor(tests[i], order=_find_order(e), test=tests)
            for i, e in enumerate(elems)]


def symbols(elem):
    return tuple(_trial_functions(elem) + _test_functions(elem))


@dataclass(repr=False)
class Coefficient(Tensor):

    def __repr__(self):
        return "w['" + self.expr + "']"


def div(a):
    assert a.order >= 1
    return Tensor('div', a.order - 1, [a])


def grad(a):
    return Tensor('grad', a.order + 1, [a])


def dot(a, b):
    assert a.order == 1
    assert b.order == 1
    return Tensor('dot', 0, [a, b])


def ddot(a, b):
    assert a.order == 2
    assert b.order == 2
    return Tensor('ddot', 0, [a, b])


def sin(a):
    return Tensor('sin', a.order, [a])

def cos(a):
    return Tensor('cos', a.order, [a])

def tan(a):
    return Tensor('tan', a.order, [a])


def trace(a):
    assert a.order == 2
    return Tensor('trace', 0, [a])

tr = trace

def eye(w, n):
    return Tensor('eye', 2, [w, Tensor(str(n))])


def mul(a, b):
    assert ((a.order == 2 and b.order == 1)
            or (a.order == 2 and b.order == 2))
    return Tensor('mul', b.order, [a, b])
