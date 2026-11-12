"""Compile forms into multithreaded C assemblers (experimental).

:class:`CBilinearForm` and :class:`CLinearForm` are drop-in replacements for
:class:`~skfem.assembly.BilinearForm` and :class:`~skfem.assembly.LinearForm`:

>>> from skfem import MeshTri, Basis, ElementTriP2
>>> from skfem.helpers import dot, grad
>>> from skfem.experimental.cforms import CBilinearForm
>>> @CBilinearForm(nthreads=4, cache_sparsity=True)
... def laplace(u, v, w):
...     return dot(grad(u), grad(v))
>>> A = laplace.assemble(Basis(MeshTri().refined(3), ElementTriP2()))

How it works:

1. *Tracing.*  The form is called once with symbolic arguments: ``u``, ``v``
   and the entries of ``w`` are :class:`~skfem.element.DiscreteField` objects
   whose entries are :class:`CExpr` instances (C expressions) instead of
   floats.  The usual :mod:`skfem.helpers` (``dot``, ``grad``, ``sym_grad``,
   ``eye``, ...) are based on ``np.einsum``, indexing and arithmetic, which
   all work on NumPy object arrays, so the same helpers build the C
   expression of the integrand.  Nothing in :mod:`skfem.helpers` is changed.
   The expression is a graph where identical subexpressions are shared, and
   every operation is evaluated at the outermost loop it can be hoisted to:
   once per quadrature point (depends only on ``w``), once per test or
   trial function, or in the innermost loop.

   The basis functions are split into groups sharing the field of
   :class:`~skfem.element.ElementComposite`, the reference element and the
   nonzero component of :class:`~skfem.element.ElementVector`.  The form is
   traced for each pair of a test and a trial group with the other fields
   and components set to zero, so the zero blocks are simplified away and
   pairs which do not interact are skipped altogether.  The traces are
   cached by the structure of the bases and ``w``.

2. *Code generation.*  The reference basis functions are tabulated at the
   quadrature points in Python (via ``Element.lbasis``) and embedded into the
   C source as constant arrays.  The element loop transforms them using
   ``invDF``, ``DF`` and ``dx`` that are taken from the Basis object, so any
   mapping (affine, isoparametric) is supported without writing C for it:
   identity for :math:`H^1`, the contravariant Piola transformation for
   :math:`H(div)` and the covariant one for :math:`H(curl)` elements.  Where
   the basis depends on the orientation of the element (DOFs of reversed
   facets as in ``ElementTriP3``, or the orientation signs of
   :math:`H(div)` and :math:`H(curl)` elements), each element has a code
   selecting the reference functions and their signs from small tables.
   DOF vectors given to ``assemble`` and the global coordinates ``w.x``
   are interpolated inside the kernel.

3. *Compilation.*  The source is compiled with the system C compiler (``CC``
   environment variable, defaults to ``cc``) with POSIX threads, and loaded
   via :mod:`ctypes`.  Libraries are cached on disk by the SHA-256 of the
   source, so the compiler runs once per (form, element, quadrature).

4. *Assembly.*  By default, the C kernel fills exactly the same COO arrays
   as the Python implementation, so ``assemble``, ``elemental``,
   ``coo_data`` and :class:`~skfem.assembly.form.COOData` work unchanged.
   Each element writes to its own slots, so threads never race.

   With ``cache_sparsity=True``, the CSR sparsity pattern is computed once
   per pair of bases and cached, and :meth:`CBilinearForm.assemble` adds the
   local matrices directly to the CSR data array.  To avoid races, the
   elements are grouped so that concurrently running threads write to
   disjoint rows: compact blocks by recursive coordinate bisection first,
   then independent sets of the remaining interface elements, see
   :func:`_partition`.

Supported are :math:`H^1`, :math:`H(div)` and :math:`H(curl)` elements
using the default transformations, and ``ElementVector``, ``ElementDG`` and
``ElementComposite`` of them.  Unsupported cases fall back to the Python
implementation with a warning, e.g., facet bases, ``ElementGlobal`` and
other elements with a custom ``gbasis``, nested ``ElementVector`` or
``ElementComposite``, complex dtype, forms branching on values (``np.where``,
``np.maximum``), or no C compiler as in Pyodide.

"""
import ctypes
import hashlib
import numbers
import os
import shutil
import subprocess
import sys
import tempfile
import warnings
import weakref
from typing import Dict, Tuple

import numpy as np
from numpy import ndarray

from skfem.assembly.basis import CellBasis
from skfem.mapping import MappingAffine, MappingIsoparametric
from skfem.assembly.form import BilinearForm, LinearForm
from skfem.assembly.form.form import FormExtraParams
from skfem.element import (DiscreteField, ElementComposite, ElementDG,
                           ElementH1, ElementHcurl, ElementHdiv,
                           ElementVector)


# ---------------------------------------------------------------------------
# Symbolic scalars
# ---------------------------------------------------------------------------

def _const(x) -> float:
    if isinstance(x, complex):
        raise NotImplementedError("Complex numbers are not supported.")
    return float(x)


def _is(x, value) -> bool:
    return isinstance(x, numbers.Number) and x == value


class _Graph:
    """Expression graph of one trace; identical subexpressions are shared."""

    def __init__(self):
        self.table = {}

    def intern(self, key, make):
        node = self.table.get(key)
        if node is None:
            node = make()
            node.id = len(self.table)
            self.table[key] = node
        return node


_graph = _Graph()  # replaced by a fresh graph for every trace


def _key(x):
    return ('n', x.id) if isinstance(x, CExpr) else ('c', _const(x))


class CExpr:
    """A scalar C expression; supports the arithmetic used in forms.

    Expressions are nodes of a graph: a symbol (a C variable), or an
    operation ``op`` applied to ``args`` (CExprs or numbers).  Identical
    subexpressions are the same node, which is used for hoisting.

    """

    __slots__ = ('op', 'args', 'deps', 'id')

    @staticmethod
    def symbol(name: str, deps: frozenset = frozenset()) -> 'CExpr':
        """A C variable; ``deps`` are the loop indices it depends on."""
        def make():
            node = CExpr()
            node.op, node.args, node.deps = 'sym', (name,), deps
            return node
        return _graph.intern(('sym', name), make)

    @staticmethod
    def apply(op: str, *args) -> 'CExpr':
        if op in ('+', '*'):  # commutative: canonical order of operands
            args = tuple(sorted(args, key=_key))

        def make():
            node = CExpr()
            node.op, node.args = op, args
            node.deps = frozenset().union(
                *(a.deps for a in args if isinstance(a, CExpr)))
            return node
        return _graph.intern((op,) + tuple(_key(a) for a in args), make)

    @property
    def src(self) -> str:
        """Inline C source of the expression (for inspection)."""
        return _render(self, lambda node: node.src)

    def __repr__(self):
        return self.src

    # arithmetic, with trivial simplifications so that the zero blocks of
    # e.g. vectorial fields do not end up in the generated source
    def __add__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return self
        return CExpr.apply('+', self, other)

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return self
        return CExpr.apply('-', self, other)

    def __rsub__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return -self
        return CExpr.apply('-', other, self)

    def __mul__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return 0.
        if _is(other, 1):
            return self
        return CExpr.apply('*', self, other)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __truediv__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 1):
            return self
        return CExpr.apply('/', self, other)

    def __rtruediv__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return 0.
        return CExpr.apply('/', other, self)

    def __pow__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if isinstance(other, numbers.Integral) and 0 < other <= 4:
            out = self
            for _ in range(int(other) - 1):
                out = out * self
            return out
        return CExpr.apply('pow', self, other)

    def __rpow__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        return CExpr.apply('pow', other, self)

    def __neg__(self):
        return CExpr.apply('neg', self)

    def __pos__(self):
        return self

    def __abs__(self):
        return CExpr.apply('fabs', self)

    def __bool__(self):
        raise TypeError("Symbolic expressions have no truth value; "
                        "branching on values is not supported in C forms.")

    # NumPy calls these methods for ufuncs on object arrays, e.g. np.sqrt
    def _fun(name):
        def f(self):
            return CExpr.apply(name, self)
        return f

    sqrt = _fun('sqrt')
    exp = _fun('exp')
    log = _fun('log')
    sin = _fun('sin')
    cos = _fun('cos')
    tan = _fun('tan')
    arcsin = _fun('asin')
    arccos = _fun('acos')
    arctan = _fun('atan')
    sinh = _fun('sinh')
    cosh = _fun('cosh')
    tanh = _fun('tanh')
    absolute = _fun('fabs')
    arcsinh = _fun('asinh')
    arccosh = _fun('acosh')
    arctanh = _fun('atanh')
    log1p = _fun('log1p')
    expm1 = _fun('expm1')
    log2 = _fun('log2')
    log10 = _fun('log10')
    exp2 = _fun('exp2')
    cbrt = _fun('cbrt')
    del _fun

    def _fun2(name):
        def f(self, other):
            if not isinstance(other, (CExpr, numbers.Number)):
                return NotImplemented
            return CExpr.apply(name, self, other)
        return f

    arctan2 = _fun2('atan2')
    hypot = _fun2('hypot')
    del _fun2


def _render(node: CExpr, ref) -> str:
    """C source of one operation; ``ref`` gives the source of operands."""
    if node.op == 'sym':
        return node.args[0]
    args = [ref(a) if isinstance(a, CExpr) else repr(_const(a))
            for a in node.args]
    if node.op in ('+', '-', '*', '/'):
        return "({} {} {})".format(args[0], node.op, args[1])
    if node.op == 'neg':
        return "(-{})".format(args[0])
    return "{}({})".format(node.op, ", ".join(args))


def _emit(roots, bilinear: bool, uside=None, vside=None):
    """Generate the loops over basis functions with hoisted temporaries.

    ``roots`` maps (test group, trial group) -- or the test group for
    linear forms -- to the traced integrand.  Every operation is computed
    once at the outermost loop level allowed by its dependencies: per
    quadrature point (only 'w'), per trial function of a group (precomputed
    to arrays before the loops), per test function of a group, or in the
    innermost loop of a pair of groups.  Pairs whose integrand is zero,
    e.g. different components of vectorial fields, are skipped.

    """
    roots = {key: root for key, root in roots.items()
             if isinstance(root, CExpr) or _const(root) != 0.}

    # operations reachable from the roots, in post-order
    nodes, seen = [], set()
    for root in roots.values():
        if not isinstance(root, CExpr):
            continue
        stack = [(root, False)]
        while stack:
            node, done = stack.pop()
            if done:
                nodes.append(node)
                continue
            if node.id in seen or node.op == 'sym':
                continue
            seen.add(node.id)
            stack.append((node, True))
            for a in reversed(node.args):
                if isinstance(a, CExpr) and a.op != 'sym':
                    stack.append((a, False))

    names = {}
    qcode, jcode, icode, ijcode = [], {}, {}, {}
    jdecl = {}

    def ref(node):
        if not isinstance(node, CExpr):
            return repr(_const(node))
        return names[node.id] if node.op != 'sym' else node.args[0]

    for n, node in enumerate(nodes):
        src = _render(node, ref)
        idx = dict(node.deps)  # e.g. {'i': 'V0', 'j': 'U2'}
        if len(idx) != len(node.deps):
            raise _Unsupported("Expression mixes basis function groups.")
        if not idx:
            names[node.id] = "tq{}".format(n)
            qcode.append("    const double tq{} = {};".format(n, src))
        elif set(idx) == {'j'}:
            g = idx['j']
            names[node.id] = "tj{}[j]".format(n)
            jdecl.setdefault(g, []).append("tj{}[M_{}]".format(n, g))
            jcode.setdefault(g, []).append("      tj{}[j] = {};"
                                           .format(n, src))
        elif set(idx) == {'i'}:
            names[node.id] = "ti{}".format(n)
            icode.setdefault(idx['i'], []).append(
                "      const double ti{} = {};".format(n, src))
        else:
            names[node.id] = "tk{}".format(n)
            ijcode.setdefault((idx['i'], idx['j']), []).append(
                "          const double tk{} = {};".format(n, src))

    out = ["\n".join(qcode)]
    for g in jdecl:
        out.append("    double {};\n"
                   "    for (int64_t j = 0; j < M_{}; j++) {{\n{}\n    }}"
                   .format(", ".join(jdecl[g]), g, "\n".join(jcode[g])))
    if bilinear:
        for gv in vside:
            pairs = [gu for gu in uside if (gv, gu) in roots]
            if not pairs:
                continue
            out.append("    for (int64_t i = 0; i < M_{gv}; i++) {{\n"
                       "      const int64_t iv = IDX_{gv}[i];\n{icode}"
                       .format(gv=gv, icode="\n".join(icode.get(gv, []))))
            for gu in pairs:
                out.append(
                    "      for (int64_t j = 0; j < M_{gu}; j++) {{\n{ij}\n"
                    "        A[IDX_{gu}[j] * NV + iv] += {expr} * w_dx;\n"
                    "      }}".format(gu=gu,
                                      ij="\n".join(ijcode.get((gv, gu), [])),
                                      expr=ref(roots[(gv, gu)])))
            out.append("    }")
    else:
        for gv in vside:
            if gv not in roots:
                continue
            out.append("    for (int64_t i = 0; i < M_{gv}; i++) {{\n{icode}\n"
                       "      A[IDX_{gv}[i]] += {expr} * w_dx;\n    }}"
                       .format(gv=gv, icode="\n".join(icode.get(gv, [])),
                               expr=ref(roots[gv])))
    return "\n".join(out)


def _symbols(fmt: str, shape: Tuple[int, ...], offset: int = 0,
             deps: frozenset = frozenset()) -> ndarray:
    """Object array of CExpr's named by ``fmt.format(offset + flat_index)``.

    Two trailing axes of length one are appended so that the arrays have the
    same rank as the numerical arrays (..., nelems, nqp) seen by forms;
    shape-based helpers such as ``inner`` or ``identity`` then work as is.
    ``deps`` are the loop indices the symbols depend on.

    """
    out = np.empty(shape, dtype=object)
    for n, ix in enumerate(np.ndindex(*shape)):
        out[ix] = CExpr.symbol(fmt.format(offset + n), deps)
    return out.reshape(shape + (1, 1))


def _zeros(shape: Tuple[int, ...]) -> ndarray:
    """Object array of zeros with the trailing axes, see :func:`_symbols`."""
    out = np.empty(shape, dtype=object)
    out.fill(0.)
    return out.reshape(shape + (1, 1))


def _simplify_result(out):
    """The traced integrand: a CExpr or a number."""
    out = np.asarray(out, dtype=object)
    if out.size != 1:
        raise ValueError("The form must return a scalar, got an array of "
                         "shape {}.".format(out.shape[:-2]))
    out = out.reshape(-1)[0]
    if not isinstance(out, (CExpr, numbers.Number)):
        raise TypeError("Cannot convert '{}' to C.".format(type(out)))
    return out


class _TracingParams(FormExtraParams):
    """FormExtraParams which records the accessed keys."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.accessed = set()

    def __getitem__(self, key):
        self.accessed.add(key)
        return super().__getitem__(key)

    def __getattr__(self, attr):
        if attr in self:
            return self[attr]
        raise AttributeError("Attribute '{}' not found in 'w'.".format(attr))


# ---------------------------------------------------------------------------
# Elements
# ---------------------------------------------------------------------------

class _Unsupported(Exception):
    pass


#: derivative attribute of DiscreteField for each transformation
_DERIVATIVE = {'h1': 'grad', 'hdiv': 'div', 'hcurl': 'curl'}


class _Field:
    """A field of the form arguments.

    ``kind`` is the transformation of the reference basis: 'h1' (identity,
    scalar or ``vdim`` components), 'hdiv' (contravariant Piola) or 'hcurl'
    (covariant Piola).  An :class:`~skfem.element.ElementComposite` has a
    field for each of its elements.

    """

    def __init__(self, kind: str, vdim=None):
        self.kind, self.vdim = kind, vdim

    def shapes(self, gdim: int):
        """Shapes of the value and the derivative."""
        if self.kind == 'h1':
            v = () if self.vdim is None else (self.vdim,)
            return v, v + (gdim,)
        if self.kind == 'hdiv':
            return (gdim,), ()
        return (gdim,), (() if gdim == 2 else (gdim,))

    def make(self, value, deriv) -> DiscreteField:
        return DiscreteField(**{'value': value,
                                _DERIVATIVE[self.kind]: deriv})

    def zeros(self, gdim: int) -> DiscreteField:
        vshape, dshape = self.shapes(gdim)
        return self.make(_zeros(vshape), _zeros(dshape))

    def symbols(self, vfmt: str, dfmt: str, gdim: int, deps=frozenset(),
                comp=None) -> DiscreteField:
        """Field where the values are the C variables ``vfmt.format(n)`` and
        the derivatives ``dfmt.format(n)``; for a component ``comp`` of a
        vectorial field, the other components are zero."""
        vshape, dshape = self.shapes(gdim)
        if comp is None:
            return self.make(_symbols(vfmt, vshape, deps=deps),
                             _symbols(dfmt, dshape, deps=deps))
        value, deriv = _zeros(vshape), _zeros(dshape)
        value[comp] = _symbols(vfmt, (), deps=deps)
        deriv[comp] = _symbols(dfmt, (gdim,), deps=deps)
        return self.make(value, deriv)


class _Group:
    """Basis functions sharing the field, the reference element (``leaf``)
    and the nonzero component of a vectorial field (``comp``).

    ``members`` lists ``(i, s)``: the index ``i`` of the basis function in
    the element and the index ``s`` of the reference function in ``leaf``.

    """

    def __init__(self, leaf, kind, field, comp, members):
        self.leaf, self.kind = leaf, kind
        self.field, self.comp = field, comp
        self.members = members

    def sizes(self, ldim: int, gdim: int):
        """Number of reference values and derivatives, and the same for the
        global basis functions."""
        if self.kind == 'h1':
            return 1, ldim, 1, gdim
        if self.kind == 'hdiv':
            return ldim, 1, gdim, 1
        return ldim, 1 if ldim == 2 else ldim, gdim, 1 if gdim == 2 else gdim


def _describe(elem, nbfun: int):
    """Split the basis functions of ``elem`` into fields and groups."""
    if isinstance(elem, ElementComposite):
        where = [tuple(int(a) for a in elem._deduce_bfun(i))
                 for i in range(nbfun)]
        fields, groups = [], []
        for n, sub in enumerate(elem.elems):
            if isinstance(sub, ElementComposite):
                raise _Unsupported("Nested ElementComposite")
            local = {ind: i for i, (m, ind) in enumerate(where) if m == n}
            subfields, subgroups = _describe(sub, len(local))
            for g in subgroups:
                g.field += len(fields)
                g.members = [(local[i], s) for i, s in g.members]
            fields += subfields
            groups += subgroups
        return fields, groups
    if isinstance(elem, ElementVector):
        subfields, subgroups = _describe(elem.elem, nbfun // elem.dim)
        if (len(subfields) != 1
                or subfields[0].kind != 'h1'
                or subfields[0].vdim is not None):
            raise _Unsupported("ElementVector of " + type(elem.elem).__name__)
        d = elem.dim
        return [_Field('h1', d)], [
            _Group(g.leaf, 'h1', 0, c, [(i * d + c, s) for i, s in g.members])
            for c in range(d) for g in subgroups
        ]
    if isinstance(elem, ElementDG):
        return _describe(elem.elem, nbfun)
    for kind, base in (('h1', ElementH1),
                       ('hdiv', ElementHdiv),
                       ('hcurl', ElementHcurl)):
        if isinstance(elem, base):
            if type(elem).gbasis is not base.gbasis:
                raise _Unsupported("Custom gbasis in " + type(elem).__name__)
            return [_Field(kind)], [
                _Group(elem, kind, 0, None, [(i, i) for i in range(nbfun)])
            ]
    raise _Unsupported(type(elem).__name__)


def _orientation(group, mapping, tind, s: int):
    """Reference function and multiplier of the function ``s`` per element.

    Mirrors :meth:`~skfem.element.Element._oriented_lbasis`, which swaps the
    DOFs of reversed facets, and ``orient`` of H(div) and H(curl) elements.

    """
    leaf = group.leaf
    t = mapping.mesh.t if tind is None else mapping.mesh.t[:, tind]
    src = np.full(t.shape[1], s, dtype=np.int64)
    mult = np.ones(t.shape[1])
    offset = leaf.refdom.nnodes * leaf.nodal_dofs
    if (leaf.dim == 2
            and leaf.facet_dofs >= 2
            and offset <= s < offset + leaf.refdom.nfacets * leaf.facet_dofs):
        facet, j = divmod(s - offset, leaf.facet_dofs)
        rev = leaf.reversed_facet_dof(j)
        if rev is not None:
            a, b = leaf.refdom.facets[facet]
            reverse = t[a] > t[b]
            mult = np.where(reverse, rev[1], 1.)
            if rev[0] != j:
                src = np.where(reverse,
                               offset + facet * leaf.facet_dofs + rev[0], s)
    if group.kind != 'h1':
        mult = mult * leaf.orient(mapping, s, tind)
        if group.kind == 'hdiv' and np.any(np.abs(mult) != 1.):
            raise _Unsupported("Non-unit orientation in H(div)")
    return src, mult


def _compress(S: ndarray, M: ndarray):
    """Number the distinct columns of (S, M); returns the number of each
    column and a representative column for each number."""
    nt = S.shape[1]
    if nt == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(1, dtype=np.int64)
    varying = np.nonzero((S != S[:, :1]).any(axis=1)
                         | (M != M[:, :1]).any(axis=1))[0]
    code = np.zeros(nt, dtype=np.int64)
    radix = 1
    for r in varying:
        _, inv = np.unique(np.vstack((S[r], M[r])), axis=1,
                           return_inverse=True)
        n = int(inv.max()) + 1
        if radix * n >= 2 ** 62:  # too many combinations for an integer
            _, rep, codes = np.unique(np.vstack((S[varying], M[varying])).T,
                                      axis=0, return_index=True,
                                      return_inverse=True)
            return codes.reshape(-1).astype(np.int64), rep
        code += inv.reshape(-1) * radix
        radix *= n
    _, rep, codes = np.unique(code, return_index=True, return_inverse=True)
    return codes.reshape(-1).astype(np.int64), rep


class _Side:
    """The basis functions of a Basis, for code generation."""

    def __init__(self, basis):
        X = basis.X
        self.ldim, self.gdim = X.shape[0], basis.mesh.p.shape[0]
        self.fields, self.groups = _describe(basis.elem, basis.Nbfun)
        if (any(g.kind != 'h1' for g in self.groups)
                and self.ldim != self.gdim):
            raise _Unsupported("H(div) or H(curl) on a manifold")
        srcs, mults = [], []
        for g in self.groups:
            for _, s in g.members:
                src, mult = _orientation(g, basis.mapping, basis.tind, s)
                srcs.append(src)
                mults.append(mult)
        S, M = np.array(srcs), np.array(mults)
        self.codes, rep = _compress(S, M)
        self.ncodes = len(rep)
        pos = 0
        for g in self.groups:
            n = len(g.members)
            src, g.mult = S[pos:pos + n, rep].T, M[pos:pos + n, rep].T
            pos += n
            g.rows = sorted(set(src.ravel().tolist())
                            | {s for _, s in g.members})
            row = {s: r for r, s in enumerate(g.rows)}
            g.src = np.vectorize(row.get, otypes=[np.int64])(src)
            nvr, ndr, _, _ = g.sizes(self.ldim, self.gdim)
            nq = X.shape[-1]
            g.phi = np.empty((len(g.rows), nvr, nq))
            g.dphi = np.empty((len(g.rows), ndr, nq))
            for r, s in enumerate(g.rows):
                phi, dphi = g.leaf.lbasis(X, s)
                g.phi[r] = np.broadcast_to(phi, (nvr, nq))
                g.dphi[r] = np.broadcast_to(dphi, (ndr, nq))

    @property
    def structure(self):
        """What the traced form sees of this basis."""
        return (tuple((f.kind, f.vdim) for f in self.fields),
                tuple((g.kind, g.field, g.comp, len(g.members))
                      for g in self.groups),
                self.ldim, self.gdim)

    @property
    def needs_DF(self) -> bool:
        return any(g.kind != 'h1' for g in self.groups)

    def arguments(self, name: str, gdim: int):
        """Symbolic form arguments for each group: the fields of the group's
        basis functions are C variables, the other fields are zero."""
        index = 'j' if name == 'U' else 'i'
        out = {}
        for n, g in enumerate(self.groups):
            G = name + str(n)
            deps = frozenset([(index, G)])
            out[G] = [
                f.symbols("{}v[{}][{{}}]".format(G, index),
                          "{}d[{}][{{}}]".format(G, index),
                          gdim, deps, g.comp) if m == g.field
                else f.zeros(gdim)
                for m, f in enumerate(self.fields)
            ]
        return out

    def tables(self, name: str) -> str:
        """Constant tables of the groups for the C source."""
        memo = self.__dict__.setdefault('_tables', {})
        if name not in memo:
            memo[name] = self._make_tables(name)
        return memo[name]

    def _make_tables(self, name: str) -> str:
        src = "\n#define NCODE_{} {}\n".format(name, self.ncodes)
        for n, g in enumerate(self.groups):
            G = name + str(n)
            src += ("#define M_{G} {m}\n"
                    "static const int64_t IDX_{G}[M_{G}] = {idx};\n"
                    "static const int64_t SRC_{G}[NCODE_{S}][M_{G}] = {src};\n"
                    "static const double MUL_{G}[NCODE_{S}][M_{G}] = {mul};\n"
                    "static const double PHI_{G}[{r}][{nv}][NQ] = {phi};\n"
                    "static const double DPHI_{G}[{r}][{nd}][NQ] = {dphi};\n"
                    .format(G=G, S=name, m=len(g.members),
                            idx=_carray(np.array([i for i, _ in g.members])),
                            src=_carray(g.src), mul=_carray(g.mult),
                            r=len(g.rows), nv=g.phi.shape[1],
                            nd=g.dphi.shape[1], phi=_carray(g.phi),
                            dphi=_carray(g.dphi)))
        return src

    def declarations(self, name: str) -> str:
        out = ["  const int64_t code{} = in->code{}[k];".format(name, name)]
        for n, g in enumerate(self.groups):
            _, _, nvp, ndp = g.sizes(self.ldim, self.gdim)
            out.append("  double {G}v[M_{G}][{nv}], {G}d[M_{G}][{nd}];"
                       .format(G=name + str(n), nv=nvp, nd=ndp))
        return "\n".join(out)

    def evaluation(self, name: str) -> str:
        """C code evaluating the global basis functions at point q."""
        out = []
        for n, g in enumerate(self.groups):
            G = name + str(n)
            body = _EVAL[g.kind]
            if g.kind == 'hcurl':
                body = body.format(curl=_CURL2D if self.gdim == 2
                                   else _CURL3D)
            out.append(_EVAL_LOOP.format(G=G, S=name, kind=g.kind,
                                         body=body.format(G=G)))
        return "\n".join(out)


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _carray(a: ndarray) -> str:
    """Nested C initializer list."""
    if a.ndim == 0:
        return repr(a.item())
    return "{" + ", ".join(_carray(x) for x in a) + "}"


_EVAL_LOOP = """
    /* basis functions of group {G} ({kind}) at quadrature point q */
    for (int64_t m = 0; m < M_{G}; m++) {{
      const int64_t r = SRC_{G}[code{S}][m];
      const double c = MUL_{G}[code{S}][m];
{body}
    }}"""


# global basis functions from the reference ones; J is invDF, DFq is DF
_EVAL = {
    'h1': """      {G}v[m][0] = c * PHI_{G}[r][0][q];
      for (int g = 0; g < GDIM; g++) {{
        double s = 0.;
        for (int a = 0; a < LDIM; a++)
          s += J[a * GDIM + g] * DPHI_{G}[r][a][q];
        {G}d[m][g] = c * s;
      }}""",
    'hdiv': """      for (int g = 0; g < GDIM; g++) {{
        double s = 0.;
        for (int a = 0; a < LDIM; a++)
          s += DFq[g * LDIM + a] * PHI_{G}[r][a][q];
        {G}v[m][g] = c * s / fabs(detq);
      }}
      {G}d[m][0] = c * DPHI_{G}[r][0][q] / fabs(detq);""",
    'hcurl': """      for (int g = 0; g < GDIM; g++) {{{{
        double s = 0.;
        for (int a = 0; a < LDIM; a++)
          s += J[a * GDIM + g] * PHI_{{G}}[r][a][q];
        {{G}}v[m][g] = c * s;
      }}}}
{curl}""",
}

_CURL2D = """      {G}d[m][0] = c * DPHI_{G}[r][0][q] / detq;"""

_CURL3D = """      for (int g = 0; g < GDIM; g++) {{
        double s = 0.;
        for (int a = 0; a < LDIM; a++)
          s += DFq[g * LDIM + a] * DPHI_{G}[r][a][q];
        {G}d[m][g] = c * s / detq;
      }}"""


_DF_TEMPLATE = """
    /* DF (strided) and its determinant at quadrature point q */
    double DFq[GDIM * LDIM];
    for (int a = 0; a < GDIM; a++)
      for (int b = 0; b < LDIM; b++)
        DFq[a * LDIM + b] = in->DF[a * in->dfs[0] + b * in->dfs[1]
                                   + k * in->dfs[2] + q * in->dfs[3]];
    const double detq = {det};
    (void)detq;"""

_DET = {
    1: "DFq[0]",
    2: "DFq[0] * DFq[3] - DFq[1] * DFq[2]",
    3: ("DFq[0] * (DFq[4] * DFq[8] - DFq[5] * DFq[7])"
        " - DFq[1] * (DFq[3] * DFq[8] - DFq[5] * DFq[6])"
        " + DFq[2] * (DFq[3] * DFq[7] - DFq[4] * DFq[6])"),
}


_GATHER_TEMPLATE = """
  /* local coefficients of the DOF vector '{name}' of 'w' */
  double C{f}[N{B}];
  for (int64_t j = 0; j < N{B}; j++)
    C{f}[j] = in->coef[{f} * in->coef_n + in->idofs[j * in->nt + k]];"""


_INTERP_DECL = """
    /* '{name}' of 'w', field {n}, interpolated at quadrature point q */
    double F{f}x{n}v[{nv}], F{f}x{n}d[{nd}];
    memset(F{f}x{n}v, 0, sizeof(F{f}x{n}v));
    memset(F{f}x{n}d, 0, sizeof(F{f}x{n}d));"""

_INTERP_GROUP = """
    for (int64_t m = 0; m < M_{G}; m++) {{
      const double cc = C{f}[IDX_{G}[m]];
      for (int n = 0; n < {nv}; n++)
        F{f}x{fld}v[{voff} + n] += cc * {G}v[m][n];
      for (int n = 0; n < {nd}; n++)
        F{f}x{fld}d[{doff} + n] += cc * {G}d[m][n];
    }}"""


_KERNEL_TEMPLATE = """
#include <math.h>
#include <stdint.h>
#include <string.h>
#include <pthread.h>

#define NQ {nqp}
#define LDIM {ldim}
#define GDIM {gdim}
#define NV {nv}
#define NU {nu}
#define NQD {nqd}
typedef {idx_t} idx_t;
{tables}

/* inputs shared by all threads */
struct input {{
  int64_t nt;
  const double *invDF;   /* inverse Jacobians (LDIM, GDIM, nt, NQ) ... */
  const int64_t *js;     /* ... with strides js[0..3] (in doubles) */
  const double *dx;      /* quadrature weights times |det DF| (nt, NQ) */
  const int64_t *dxs;    /* ... with strides dxs[0..1] (in doubles) */
  const double *qd;      /* (nt, NQ, NQD) fields of 'w' at quad. points */
  const double *P;       /* scalar parameters of 'w' */
  const double *coef;    /* DOF vectors of 'w', (ncoef, coef_n) */
  int64_t coef_n;
  const int64_t *idofs;  /* element DOFs of the interpolation basis */
  const double *gp;      /* geometry nodes (GDIM, gnp) for 'x' ... */
  int64_t gnp;
  const int64_t *gconn;  /* ... and their element connectivity (NG, nt) */
  const double *DF;      /* Jacobians (GDIM, LDIM, nt, NQ) for H(div) and
                            H(curl) ... */
  const int64_t *dfs;    /* ... with strides dfs[0..3] (in doubles) */
  const int64_t *codeU;  /* orientation code of each element, selects the */
  const int64_t *codeV;  /* reference functions and signs of the tables */
  const idx_t *indptr;   /* CSR sparsity pattern (direct assembly) */
  const idx_t *indices;
  const int64_t *udofs;  /* (NU, nt) element DOFs of trial functions */
  const int64_t *vdofs;  /* (NV, nt) element DOFs of test functions */
  const int64_t *perm;   /* element order, grouped by thread */
}};

/* range of elements handled by one thread */
struct task {{
  const struct input *in;
  int64_t start, stop;
  double *out;
}};

/* local matrix A[j * NV + i] (or local vector A[i]) of element k */
static void local(const struct input *restrict in, int64_t k,
                  double *restrict A)
{{
{decls}
  const double *restrict P = in->P;
  (void)P;
{gather}
  memset(A, 0, sizeof(double) * NU * NV);
  for (int64_t q = 0; q < NQ; q++) {{
    /* strided access: e.g. affine mappings have zero stride over q */
    double J[LDIM * GDIM];
    for (int a = 0; a < LDIM; a++)
      for (int g = 0; g < GDIM; g++)
        J[a * GDIM + g] = in->invDF[a * in->js[0] + g * in->js[1]
                                    + k * in->js[2] + q * in->js[3]];
    const double *Q = in->qd + (k * NQ + q) * NQD;
    const double w_dx = in->dx[k * in->dxs[0] + q * in->dxs[1]];
    (void)Q;
{eval}
{interp}
{loops}
  }}
}}

/* write to the COO arrays of scikit-fem; every element has own slots */
static void *worker_coo(void *arg)
{{
  const struct task *t = arg;
  const int64_t nt = t->in->nt;
  double A[NU * NV];
  for (int64_t k = t->start; k < t->stop; k++) {{
    local(t->in, k, A);
    for (int64_t l = 0; l < NU * NV; l++)
      t->out[l * nt + k] = A[l];
  }}
  return NULL;
}}

/* add local matrices of perm[start..stop] directly to CSR data */
static void add_csr(const struct input *restrict in, int64_t start,
                    int64_t stop, double *restrict out)
{{
  const int64_t nt = in->nt;
  double A[NU * NV];
  for (int64_t e = start; e < stop; e++) {{
    const int64_t k = in->perm[e];
    local(in, k, A);
    for (int64_t i = 0; i < NV; i++) {{
      const int64_t row = in->vdofs[i * nt + k];
      for (int64_t j = 0; j < NU; j++) {{
        const idx_t col = (idx_t)in->udofs[j * nt + k];
        /* binary search in the sorted column indices of the row */
        int64_t lo = in->indptr[row], hi = in->indptr[row + 1] - 1;
        while (lo < hi) {{
          const int64_t mid = lo + (hi - lo) / 2;
          if (in->indices[mid] < col) lo = mid + 1; else hi = mid;
        }}
        out[lo] += A[j * NV + i];
      }}
    }}
  }}
}}

/* portable barrier (pthread_barrier_t is optional in POSIX) */
struct barrier {{
  pthread_mutex_t mutex;
  pthread_cond_t cond;
  int64_t n, count, generation;
}};

static void barrier_wait(struct barrier *b)
{{
  pthread_mutex_lock(&b->mutex);
  const int64_t generation = b->generation;
  if (++b->count == b->n) {{
    b->count = 0;
    b->generation++;
    pthread_cond_broadcast(&b->cond);
  }} else {{
    while (generation == b->generation)
      pthread_cond_wait(&b->cond, &b->mutex);
  }}
  pthread_mutex_unlock(&b->mutex);
}}

/* thread 'id' assembles group phases[f] + id of every phase f */
struct csr_task {{
  const struct input *in;
  const int64_t *parts, *phases;
  int64_t nphases;
  int64_t id;
  int64_t tail;  /* also handles ids tail, tail + 1, .., nthreads - 1 */
  int64_t nthreads;
  struct barrier *bar;
  double *out;
}};

static void *worker_csr(void *arg)
{{
  const struct csr_task *t = arg;
  for (int64_t f = 0; f < t->nphases; f++) {{
    for (int64_t id = t->id; id < t->nthreads;
         id = id == t->id ? t->tail : id + 1) {{
      const int64_t g = t->phases[f] + id;
      if (g < t->phases[f + 1])
        add_csr(t->in, t->parts[g], t->parts[g + 1], t->out);
    }}
    barrier_wait(t->bar);
  }}
  return NULL;
}}

static void run(void *(*worker)(void *), struct task *tasks, int64_t n)
{{
  pthread_t threads[n];
  int created[n];
  for (int64_t p = 0; p < n; p++) {{
    created[p] = n > 1
      && pthread_create(&threads[p], NULL, worker, &tasks[p]) == 0;
    if (!created[p]) worker(&tasks[p]);  /* serial fallback */
  }}
  for (int64_t p = 0; p < n; p++)
    if (created[p]) pthread_join(threads[p], NULL);
}}

void assemble_coo(int64_t nt,
                  const double *invDF,
                  const int64_t *js,
                  const double *dx,
                  const int64_t *dxs,
                  const double *qd,
                  const double *P,
                  const double *coef,
                  int64_t coef_n,
                  const int64_t *idofs,
                  const double *gp,
                  int64_t gnp,
                  const int64_t *gconn,
                  const double *DF,
                  const int64_t *dfs,
                  const int64_t *codeU,
                  const int64_t *codeV,
                  double *data,
                  int64_t nthreads)
{{
  struct input in = {{.nt = nt, .invDF = invDF, .js = js, .dx = dx,
                      .dxs = dxs, .qd = qd, .P = P, .coef = coef,
                      .coef_n = coef_n, .idofs = idofs, .gp = gp,
                      .gnp = gnp, .gconn = gconn, .DF = DF, .dfs = dfs,
                      .codeU = codeU, .codeV = codeV}};
  if (nthreads > nt) nthreads = nt;
  if (nthreads < 1) nthreads = 1;
  struct task tasks[nthreads];
  for (int64_t p = 0; p < nthreads; p++) {{
    tasks[p].in = &in;
    tasks[p].start = nt * p / nthreads;
    tasks[p].stop = nt * (p + 1) / nthreads;
    tasks[p].out = data;
  }}
  run(worker_coo, tasks, nthreads);
}}

/* group g is perm[parts[g]..parts[g + 1]]; phase f consists of groups
   phases[f]..phases[f + 1] which write to disjoint rows and are assembled
   in parallel; there is a barrier between the phases */
void assemble_csr(int64_t nt,
                  const double *invDF,
                  const int64_t *js,
                  const double *dx,
                  const int64_t *dxs,
                  const double *qd,
                  const double *P,
                  const double *coef,
                  int64_t coef_n,
                  const int64_t *idofs,
                  const double *gp,
                  int64_t gnp,
                  const int64_t *gconn,
                  const double *DF,
                  const int64_t *dfs,
                  const int64_t *codeU,
                  const int64_t *codeV,
                  const idx_t *indptr,
                  const idx_t *indices,
                  const int64_t *udofs,
                  const int64_t *vdofs,
                  const int64_t *perm,
                  const int64_t *parts,
                  const int64_t *phases,
                  int64_t nphases,
                  double *data)
{{
  struct input in = {{.nt = nt, .invDF = invDF, .js = js, .dx = dx,
                      .dxs = dxs, .qd = qd, .P = P, .coef = coef,
                      .coef_n = coef_n, .idofs = idofs, .gp = gp,
                      .gnp = gnp, .gconn = gconn, .DF = DF, .dfs = dfs,
                      .codeU = codeU, .codeV = codeV, .indptr = indptr,
                      .indices = indices, .udofs = udofs, .vdofs = vdofs,
                      .perm = perm}};
  int64_t nthreads = 1;
  for (int64_t f = 0; f < nphases; f++)
    if (phases[f + 1] - phases[f] > nthreads)
      nthreads = phases[f + 1] - phases[f];

  struct barrier bar;
  pthread_mutex_init(&bar.mutex, NULL);
  pthread_cond_init(&bar.cond, NULL);
  bar.count = 0;
  bar.generation = 0;

  pthread_t threads[nthreads];
  struct csr_task tasks[nthreads];
  for (int64_t p = 0; p < nthreads; p++) {{
    tasks[p].in = &in;
    tasks[p].parts = parts;
    tasks[p].phases = phases;
    tasks[p].nphases = nphases;
    tasks[p].id = p;
    tasks[p].tail = nthreads;  /* no extra ids */
    tasks[p].nthreads = nthreads;
    tasks[p].bar = &bar;
    tasks[p].out = data;
  }}

  /* hold the mutex so that no thread passes the first barrier before the
     number of running threads is known */
  pthread_mutex_lock(&bar.mutex);
  int64_t created = 0;
  while (created + 1 < nthreads
         && pthread_create(&threads[created + 1], NULL, worker_csr,
                           &tasks[created + 1]) == 0)
    created++;
  bar.n = created + 1;
  pthread_mutex_unlock(&bar.mutex);

  /* the calling thread is thread 0 and takes over the ids of the threads
     that could not be created */
  tasks[0].tail = created + 1;
  worker_csr(&tasks[0]);

  for (int64_t p = 1; p <= created; p++)
    pthread_join(threads[p], NULL);
  pthread_mutex_destroy(&bar.mutex);
  pthread_cond_destroy(&bar.cond);
}}
"""


_GEOMETRY_TEMPLATE = """
    /* global coordinates 'x' at quadrature point q */
    double XQ[GDIM];
    for (int g = 0; g < GDIM; g++) XQ[g] = 0.;
    for (int64_t n = 0; n < NG; n++) {{
      const int64_t node = in->gconn[n * in->nt + k];
      for (int g = 0; g < GDIM; g++)
        XQ[g] += in->gp[g * in->gnp + node] * GPHI[n][q];
    }}"""


def _cache_dir() -> str:
    path = os.environ.get(
        'SKFEM_CFORMS_CACHE',
        os.path.join(os.path.expanduser('~'), '.cache', 'skfem', 'cforms'),
    )
    os.makedirs(path, exist_ok=True)
    return path


_libraries: Dict[str, ctypes.CDLL] = {}


def _compiler_available() -> bool:
    """Whether forms can be compiled, e.g., not in Pyodide."""
    if sys.platform == 'emscripten':
        return False
    return shutil.which(os.environ.get('CC', 'cc')) is not None


# -fno-math-errno: lets the compiler hoist exp(), sqrt(), ... of 'w' out of
# the loops over basis functions; errno is never read
_CFLAGS = ['-O3', '-fno-math-errno', '-fPIC', '-shared', '-std=c99',
           '-pthread']


def compile_source(source: str, recompile: bool = False) -> ctypes.CDLL:
    """Compile C source into a shared library (cached by hash) and load.

    The hash covers the source, the compiler and its flags.

    """
    cc = os.environ.get('CC', 'cc')
    digest = hashlib.sha256(
        "\n".join([cc] + _CFLAGS + [source]).encode('utf-8')
    ).hexdigest()[:32]
    if digest in _libraries and not recompile:
        return _libraries[digest]
    libpath = os.path.join(_cache_dir(), 'skfem_{}.so'.format(digest))
    if recompile or not os.path.isfile(libpath):
        with tempfile.TemporaryDirectory() as tmp:
            src = os.path.join(tmp, 'kernel.c')
            out = os.path.join(tmp, 'kernel.so')
            with open(src, 'w') as f:
                f.write(source)
            cmd = [cc] + _CFLAGS + [src, '-o', out, '-lm']
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                raise RuntimeError("Compiling the form failed:\n"
                                   + res.stderr)
            os.replace(out, libpath)  # atomic, safe for concurrent builds
    lib = ctypes.CDLL(libpath)
    _libraries[digest] = lib
    return lib


def _kernel_lib(source: str) -> ctypes.CDLL:
    """Compile and load an assembly kernel."""
    lib = compile_source(source)
    dptr = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    iptr = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    xptr = np.ctypeslib.ndpointer(flags='C_CONTIGUOUS')  # idx_t
    i64 = ctypes.c_int64
    vptr = ctypes.c_void_p  # strided arrays
    lib.assemble_coo.argtypes = (i64, vptr, iptr, vptr, iptr, dptr, dptr,
                                 dptr, i64, iptr, dptr, i64, iptr, vptr,
                                 iptr, iptr, iptr, dptr, i64)
    lib.assemble_coo.restype = None
    lib.assemble_csr.argtypes = (i64, vptr, iptr, vptr, iptr, dptr, dptr,
                                 dptr, i64, iptr, dptr, i64, iptr, vptr,
                                 iptr, iptr, iptr, xptr, xptr, iptr, iptr,
                                 iptr, iptr, iptr, i64, dptr)
    lib.assemble_csr.restype = None
    return lib


_PATTERN_SOURCE = """
#include <stdint.h>
#include <stdlib.h>
#include <pthread.h>

/* CSR sparsity pattern of a bilinear form, computed row by row: the
   columns of a row are gathered from the elements containing the row and
   deduplicated with a marker array, so the list of duplicated (row, col)
   pairs (nbfun^2 * nelems entries) is never formed */

struct args {
  int64_t nbu, nt, ncol, r0, r1, wide;
  const int64_t *udofs, *r2e_ptr, *r2e, *indptr;
  int64_t *count;
  void *indices;  /* int32_t or int64_t, see 'wide' */
};

#define INSERT(T) do {                                   \\
    T *o = (T *)a->indices + a->indptr[r];               \\
    int64_t i = n;                                       \\
    while (i > 0 && o[i - 1] > c) { o[i] = o[i - 1]; i--; } \\
    o[i] = (T)c;                                         \\
  } while (0)

/* count (indices == NULL) or fill and sort the columns of rows r0..r1 */
static void *rows(void *p)
{
  const struct args *a = p;
  int64_t *marker = malloc(sizeof(int64_t) * (a->ncol > 0 ? a->ncol : 1));
  if (marker == NULL) return p;  /* reported as failure */
  for (int64_t c = 0; c < a->ncol; c++) marker[c] = -1;
  for (int64_t r = a->r0; r < a->r1; r++) {
    int64_t n = 0;
    for (int64_t q = a->r2e_ptr[r]; q < a->r2e_ptr[r + 1]; q++) {
      const int64_t e = a->r2e[q];
      for (int64_t j = 0; j < a->nbu; j++) {
        const int64_t c = a->udofs[j * a->nt + e];
        if (marker[c] == r) continue;
        marker[c] = r;
        if (a->indices != NULL) {  /* insertion sort; rows are short */
          if (a->wide) INSERT(int64_t); else INSERT(int32_t);
        }
        n++;
      }
    }
    if (a->indices == NULL) a->count[r] = n;
  }
  free(marker);
  return NULL;
}

/* returns 0 on success */
static int run(const struct args *base, int64_t nrow, int64_t nthreads)
{
  if (nthreads < 1) nthreads = 1;
  if (nthreads > nrow) nthreads = nrow > 0 ? nrow : 1;
  pthread_t th[nthreads];
  struct args a[nthreads];
  int created[nthreads];
  int fail = 0;
  for (int64_t p = 0; p < nthreads; p++) {
    a[p] = *base;
    a[p].r0 = nrow * p / nthreads;
    a[p].r1 = nrow * (p + 1) / nthreads;
    created[p] = nthreads > 1
      && pthread_create(&th[p], NULL, rows, &a[p]) == 0;
    if (!created[p]) fail |= rows(&a[p]) != NULL;  /* serial fallback */
  }
  for (int64_t p = 0; p < nthreads; p++) {
    void *ret = NULL;
    if (created[p]) pthread_join(th[p], &ret);
    fail |= ret != NULL;
  }
  return fail;
}

/* step 1: row -> element map (r2e_ptr: nrow + 1, r2e: nbv * nt) and
   row pointers 'indptr' (nrow + 1); returns nnz, or -1 on failure */
int64_t pattern_count(int64_t nrow, int64_t ncol, int64_t nbu, int64_t nbv,
                      int64_t nt, const int64_t *udofs,
                      const int64_t *vdofs, int64_t *r2e_ptr, int64_t *r2e,
                      int64_t *indptr, int64_t nthreads)
{
  for (int64_t r = 0; r <= nrow; r++) r2e_ptr[r] = 0;
  for (int64_t i = 0; i < nbv * nt; i++) r2e_ptr[vdofs[i] + 1]++;
  for (int64_t r = 0; r < nrow; r++) r2e_ptr[r + 1] += r2e_ptr[r];
  int64_t *pos = malloc(sizeof(int64_t) * (nrow > 0 ? nrow : 1));
  if (pos == NULL) return -1;
  for (int64_t r = 0; r < nrow; r++) pos[r] = r2e_ptr[r];
  for (int64_t e = 0; e < nt; e++)
    for (int64_t i = 0; i < nbv; i++)
      r2e[pos[vdofs[i * nt + e]]++] = e;
  free(pos);
  struct args a = {nbu, nt, ncol, 0, 0, 0, udofs, r2e_ptr, r2e, NULL,
                   indptr + 1, NULL};
  indptr[0] = 0;
  if (run(&a, nrow, nthreads)) return -1;
  for (int64_t r = 0; r < nrow; r++) indptr[r + 1] += indptr[r];
  return indptr[nrow];
}

/* step 2: sorted column indices, int64 if 'wide' else int32;
   returns 0 on success */
int pattern_fill(int64_t nrow, int64_t ncol, int64_t nbu, int64_t nt,
                 const int64_t *udofs, const int64_t *r2e_ptr,
                 const int64_t *r2e, const int64_t *indptr, void *indices,
                 int64_t wide, int64_t nthreads)
{
  struct args a = {nbu, nt, ncol, 0, 0, wide, udofs, r2e_ptr, r2e, indptr,
                   NULL, indices};
  return run(&a, nrow, nthreads);
}
"""


def _pattern_lib() -> ctypes.CDLL:
    lib = compile_source(_PATTERN_SOURCE)
    iptr = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    xptr = np.ctypeslib.ndpointer(flags='C_CONTIGUOUS')
    i64 = ctypes.c_int64
    lib.pattern_count.argtypes = (i64, i64, i64, i64, i64, iptr, iptr, iptr,
                                  iptr, iptr, i64)
    lib.pattern_count.restype = i64
    lib.pattern_fill.argtypes = (i64, i64, i64, i64, iptr, iptr, iptr, iptr,
                                 xptr, i64, i64)
    lib.pattern_fill.restype = ctypes.c_int
    return lib


def sparsity_pattern(udofs: ndarray, vdofs: ndarray, ncol: int, nrow: int,
                     nthreads: int = 1,
                     dtype=None) -> Tuple[ndarray, ndarray]:
    """CSR pattern ``(indptr, indices)`` from element DOF arrays.

    The column indices of each row are sorted.  The index dtype is int32 if
    possible, else int64 (as in SciPy), unless ``dtype`` is given.

    """
    lib = _pattern_lib()
    nbu, nt = udofs.shape
    nbv = vdofs.shape[0]
    udofs = np.ascontiguousarray(udofs, dtype=np.int64)
    vdofs = np.ascontiguousarray(vdofs, dtype=np.int64)
    r2e_ptr = np.empty(nrow + 1, dtype=np.int64)
    r2e = np.empty(max(nbv * nt, 1), dtype=np.int64)
    indptr = np.empty(nrow + 1, dtype=np.int64)
    nnz = lib.pattern_count(nrow, ncol, nbu, nbv, nt, udofs, vdofs, r2e_ptr,
                            r2e, indptr, nthreads)
    if nnz < 0:
        raise MemoryError("Computing the sparsity pattern failed.")
    if dtype is None:
        wide = max(nnz, nrow, ncol) >= np.iinfo(np.int32).max
    else:
        wide = np.dtype(dtype) == np.int64
    indices = np.empty(nnz, dtype=np.int64 if wide else np.int32)
    if lib.pattern_fill(nrow, ncol, nbu, nt, udofs, r2e_ptr, r2e, indptr,
                        indices, int(wide), nthreads) != 0:
        raise MemoryError("Computing the sparsity pattern failed.")
    return indptr.astype(indices.dtype), indices


# ---------------------------------------------------------------------------
# Forms
# ---------------------------------------------------------------------------

def _strided(arr) -> ndarray:
    """Float64 array without copying if possible (strides may be zero)."""
    arr = np.asarray(arr)
    if arr.dtype != np.float64 or any(st % 8 for st in arr.strides):
        arr = np.ascontiguousarray(arr, dtype=np.float64)
    return arr


_basis_cache = weakref.WeakKeyDictionary()


def _cached(basis, name, make):
    """Data derived from a Basis object, remembered while it is alive."""
    try:
        entry = _basis_cache[basis]
    except KeyError:
        entry = _basis_cache[basis] = {}
    except TypeError:  # not weak referenceable
        return make()
    if name not in entry:
        entry[name] = make()
    return entry[name]


def _dofs64(basis) -> ndarray:
    """``basis.element_dofs`` as contiguous int64."""
    return _cached(basis, 'dofs64', lambda: np.ascontiguousarray(
        basis.element_dofs, dtype=np.int64))


def _geometry(basis):
    """Data for computing the global coordinates 'x' in C.

    The mapping is written as ``x = sum_n nodes[:, conn[n, k]] phi_n(X)``;
    returns ``(phi, nodes, conn)`` with ``phi`` tabulated at the quadrature
    points, or None if the mapping is not of this form.

    """
    mapping, mesh, X = basis.mapping, basis.mesh, basis.X
    if (type(mapping) is MappingAffine
            and mesh.t.shape[0] == X.shape[0] + 1):  # simplices
        phi = np.vstack((1. - X.sum(axis=0), X))
        nodes, conn = mesh.p, mesh.t
    elif type(mapping) is MappingIsoparametric:
        conn = mesh.dofs.element_dofs
        nodes = mesh.doflocs
        phi = np.array([np.broadcast_to(mapping.elem.lbasis(X, n)[0],
                                        X.shape[-1:])
                        for n in range(conn.shape[0])])
    else:
        return None
    if nodes.shape[0] != X.shape[0]:
        return None
    if basis.tind is not None:
        conn = conn[:, basis.tind]
    return (phi,
            np.ascontiguousarray(nodes, dtype=np.float64),
            np.ascontiguousarray(conn, dtype=np.int64))


class _Lazy:
    """Entry of 'w' which is evaluated only if the form uses it."""

    def __init__(self, fn, shape=()):
        self.fn = fn
        self.shape = shape


_GEOM_X = object()  # 'x' computed inside the kernel


def _strides(arr) -> ndarray:
    """Strides in units of float64."""
    return np.array(arr.strides, dtype=np.int64) // 8


def _signature(val):
    """Structure of an entry of 'w' as seen by the traced form, or None if
    it cannot be summarized (then the trace is not cached)."""
    if val is _GEOM_X:
        return ('x',)
    if isinstance(val, numbers.Number):
        return ('P',)
    if isinstance(val, _Lazy):
        return ('L', val.shape)
    if isinstance(val, DiscreteField):
        return ('F',) + tuple(
            (attr, np.shape(val if attr == 'value' else getattr(val, attr))
             [:-2])
            for attr in ('value',) + DiscreteField._extra_attrs
            if attr == 'value' or getattr(val, attr) is not None)
    if isinstance(val, tuple):
        sigs = tuple(_signature(v) for v in val)
        return None if None in sigs else ('T',) + sigs
    return None


class _CFormMixin:
    """Shared logic of CBilinearForm and CLinearForm."""

    def _check_support(self, *bases):
        if not _compiler_available():
            raise _Unsupported("no C compiler found (set CC)")
        if np.dtype(self.dtype) != np.float64:
            raise _Unsupported("dtype {}".format(self.dtype))
        for basis in bases:
            if type(basis) is not CellBasis:
                raise _Unsupported(type(basis).__name__)

    @staticmethod
    def _side(basis) -> _Side:
        return _cached(basis, 'side', lambda: _Side(basis))

    @staticmethod
    def _pack(val, offset: int, pack: bool):
        """Symbolic version of a DiscreteField (or a tuple of them) whose
        values at quadrature points are passed in ``Q``."""
        if isinstance(val, tuple):
            out, qdata = [], []
            for v in val:
                f, q, offset = _CFormMixin._pack(v, offset, pack)
                out.append(f)
                qdata += q
            return tuple(out), qdata, offset
        syms, qdata = {}, []
        for attr in ('value',) + DiscreteField._extra_attrs:
            arr = np.asarray(val) if attr == 'value' else getattr(val, attr)
            if arr is None:
                continue
            shape = np.shape(arr)[:-2]
            syms[attr] = _symbols("Q[{}]", shape, offset)
            offset += int(np.prod(shape))
            if pack:
                arr = np.asarray(arr, dtype=np.float64)
                nt, nq = arr.shape[-2:]
                qdata.append(np.moveaxis(arr.reshape(-1, nt, nq), 0, -1))
        return DiscreteField(**syms), qdata, offset

    @classmethod
    def _build_w(cls, wdict, coefs=(), cfields=(), gdim=1, used=None):
        """Symbolic 'w'.

        Scalars become runtime parameters ``P``; DiscreteFields in ``used``
        are packed into ``Q``; DOF vectors with names in ``coefs`` are
        interpolated in C, where ``cfields`` are the fields of the
        interpolation basis.

        """
        w = _TracingParams()
        params, qdata, offset = [], [], 0
        interpolated = []
        for key in coefs:
            if used is not None and key not in used:
                continue
            f = len(interpolated)
            interpolated.append(key)
            fields = [field.symbols("F{}x{}v[{{}}]".format(f, n),
                                    "F{}x{}d[{{}}]".format(f, n), gdim)
                      for n, field in enumerate(cfields)]
            dict.__setitem__(w, key,
                             fields[0] if len(fields) == 1 else tuple(fields))
        uses_x = False
        for key, val in wdict.items():
            if isinstance(val, _Lazy):
                if used is None:  # placeholder while finding used keys
                    dict.__setitem__(w, key, DiscreteField(
                        value=_symbols("Q[{}]", val.shape)))
                    continue
                if key not in used:
                    continue
                val = val.fn()
            if val is _GEOM_X:
                dict.__setitem__(w, key, DiscreteField(
                    value=_symbols("XQ[{}]", (gdim,))))
                uses_x = uses_x or (used is not None and key in used)
            elif isinstance(val, numbers.Number):
                # runtime parameter: changing the value needs no recompile
                dict.__setitem__(w, key,
                                 CExpr.symbol("P[{}]".format(len(params))))
                params.append(float(val))
            elif (isinstance(val, DiscreteField)
                  or (isinstance(val, tuple) and len(val) > 0
                      and all(isinstance(v, DiscreteField) for v in val))):
                if used is not None and key not in used:
                    continue
                # fields are passed as values at the quadrature points
                sym, q, offset = cls._pack(val, offset, used is not None)
                qdata += q
                dict.__setitem__(w, key, sym)
            else:
                # e.g. 'idx' of asm(); keep as is
                dict.__setitem__(w, key, val)
        return w, params, qdata, offset, interpolated, uses_x

    def _trace(self, argsets, wdict, coefs=(), cfields=(), gdim=1):
        """Call the form with symbolic arguments for each pair of groups.

        The first round finds out which entries of 'w' are used, the second
        builds the final expressions; only the used fields are packed.

        """
        try:
            w = self._build_w(wdict, coefs, cfields, gdim)[0]
            for args in argsets.values():
                self.form(*args, w)
            used = set(w.accessed)
            w, params, qdata, nqd, interp, uses_x = self._build_w(
                wdict, coefs, cfields, gdim, used=used)
            roots = {key: _simplify_result(self.form(*args, w))
                     for key, args in argsets.items()}
        except (TypeError, NotImplementedError, AttributeError, ValueError,
                IndexError) as err:
            # e.g. branching on values or NumPy functions without C version;
            # genuine errors in the form are raised by the Python fallback
            raise _Unsupported("Tracing the form failed: {}".format(err))
        return roots, params, qdata, nqd, interp, uses_x, used

    def _generate(self, basis, vbasis, wdict, idx_dtype=np.int32,
                  coefs=(), geom=None):
        global _graph
        _graph = _Graph()  # fresh expression graph for this trace
        # not probed from invDF: isoparametric mappings cache only the
        # latest invDF, i.e., the one of the basis that is reused below
        gdim = basis.mesh.p.shape[0]
        ldim = basis.X.shape[0]
        vside = self._side(vbasis)
        vargs = vside.arguments('V', gdim)
        if self._bilinear:
            uside = self._side(basis)
            uargs = uside.arguments('U', gdim)
            # one trace per (test group, trial group)
            argsets = {(gv, gu): tuple(uargs[gu]) + tuple(vargs[gv])
                       for gv in vargs for gu in uargs}
        else:
            uside, uargs = None, {}
            argsets = {gv: tuple(vargs[gv]) for gv in vargs}

        # DOF vectors are interpolated using the trial basis (as in
        # scikit-fem), which is the test basis for linear forms
        B, bside = ('U', uside) if self._bilinear else ('V', vside)

        # the trace depends only on the structure of the bases and 'w'
        sigs = [_signature(val) for val in wdict.values()]
        key = None if None in sigs else (
            tuple(zip(wdict, sigs)), tuple(coefs), gdim, ldim,
            tuple(side.structure for side in (uside, vside) if side))
        cache = self.__dict__.setdefault('_trace_cache', {})
        if key in cache:
            loops, used = cache[key]
            _, params, qdata, nqd, interp, uses_x = self._build_w(
                wdict, coefs, bside.fields, gdim, used=used)
        else:
            roots, params, qdata, nqd, interp, uses_x, used = self._trace(
                argsets, wdict, coefs, bside.fields, gdim)
            loops = _emit(roots, self._bilinear, list(uargs), list(vargs))
            if key is not None:
                cache[key] = (loops, used)

        gather = "".join(_GATHER_TEMPLATE.format(name=name, f=f, B=B)
                         for f, name in enumerate(interp))
        interp_src = ""
        for f, name in enumerate(interp):
            for n, field in enumerate(bside.fields):
                vshape, dshape = field.shapes(gdim)
                interp_src += _INTERP_DECL.format(
                    name=name, f=f, n=n, nv=max(int(np.prod(vshape)), 1),
                    nd=max(int(np.prod(dshape)), 1))
            for n, g in enumerate(bside.groups):
                _, _, nvp, ndp = g.sizes(ldim, gdim)
                comp = 0 if g.comp is None else g.comp
                interp_src += _INTERP_GROUP.format(
                    G=B + str(n), f=f, fld=g.field, nv=nvp, nd=ndp,
                    voff=comp * nvp, doff=comp * ndp)
        gtab = ""
        if uses_x:
            gtab = ("\n#define NG {}\nstatic const double GPHI[NG][NQ] = {};\n"
                    .format(geom[0].shape[0], _carray(geom[0])))
            interp_src = _GEOMETRY_TEMPLATE.format() + interp_src

        sides = [('V', vside)] + ([('U', uside)] if self._bilinear else [])
        needs_DF = any(side.needs_DF for _, side in sides)
        evals = (_DF_TEMPLATE.format(det=_DET[ldim]) if needs_DF else "")
        evals += "".join(side.evaluation(name) for name, side in sides)
        decls = "\n".join(side.declarations(name) for name, side in sides)
        tables = "".join(side.tables(name) for name, side in sides)

        source = _KERNEL_TEMPLATE.format(
            nqp=basis.X.shape[-1],
            ldim=ldim,
            gdim=gdim,
            nv=vbasis.Nbfun,
            nu=basis.Nbfun if self._bilinear else 1,
            nqd=max(nqd, 1),
            idx_t={np.int32: 'int32_t',
                   np.int64: 'int64_t'}[np.dtype(idx_dtype).type],
            tables=tables + gtab,
            decls=decls,
            gather=gather,
            eval=evals,
            interp=interp_src,
            loops=loops,
        )
        return source, params, qdata, interp, uses_x, needs_DF

    def _wdict(self, ubasis, kwargs):
        """Split DOF vectors from kwargs and build the numerical 'w'."""
        kwargs = dict(kwargs)
        coefs = self._split_coefs(ubasis, kwargs)
        geom = _cached(ubasis, 'geometry', lambda: _geometry(ubasis))
        if geom is None:
            defaults = ubasis.default_parameters()
        else:  # 'x' is computed in C, 'h' only if used
            defaults = {'x': _GEOM_X, 'h': _Lazy(ubasis.mesh_parameters)}
        wdict = FormExtraParams({
            **defaults,
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })
        return coefs, wdict, geom

    def source(self, ubasis, vbasis=None, **kwargs) -> str:
        """Return the generated C source (for inspection)."""
        vbasis = ubasis if vbasis is None else vbasis
        coefs, wdict, geom = self._wdict(ubasis, kwargs)
        return self._generate(ubasis, vbasis, wdict, coefs=list(coefs),
                              geom=geom)[0]

    @property
    def _nthreads(self) -> int:
        """Number of threads; nthreads <= 0 means all cores."""
        if self.nthreads > 0:
            return int(self.nthreads)
        return os.cpu_count() or 1

    #: interpolate DOF vectors given in 'w' inside the C kernel; if False,
    #: they are interpolated by NumPy and passed at quadrature points
    interpolate_in_c = True

    def _split_coefs(self, ubasis, kwargs):
        """Remove the DOF vectors to be interpolated in C from kwargs."""
        coefs = {}
        if not self.interpolate_in_c:
            return coefs
        for key in list(kwargs):
            val = kwargs[key]
            if (type(val) is ndarray
                    and val.ndim == 1
                    and len(val) == ubasis.N
                    and not np.iscomplexobj(val)):
                coefs[key] = kwargs.pop(key)
        return coefs

    def _prepare(self, ubasis, vbasis, kwargs, idx_dtype=np.int32):
        """Generate, compile and load the kernel; collect its inputs."""
        self._check_support(ubasis, vbasis)
        if ubasis.X.shape[-1] != vbasis.X.shape[-1]:
            raise ValueError("Quadrature mismatch: trial and test functions "
                             "should have same number of integration points.")
        coefs, wdict, geom = self._wdict(ubasis, kwargs)
        source, params, qdata, interp, uses_x, needs_DF = self._generate(
            ubasis, vbasis, wdict, idx_dtype, coefs=list(coefs), geom=geom)
        lib = _kernel_lib(source)
        if interp:
            coef = np.ascontiguousarray(np.stack([coefs[k] for k in interp]),
                                        dtype=np.float64)
            idofs = _dofs64(ubasis)
        else:
            coef, idofs = np.zeros(1), np.zeros(1, dtype=np.int64)
        if uses_x:
            _, gp, gconn = geom
        else:
            gp, gconn = np.zeros((1, 1)), np.zeros(1, dtype=np.int64)
        # passed with strides; e.g. affine invDF is a broadcast view
        invDF = _strided(ubasis.mapping.invDF(ubasis.X, tind=ubasis.tind))
        dx = _strided(ubasis.dx)
        DF = (_strided(ubasis.mapping.DF(ubasis.X, tind=ubasis.tind))
              if needs_DF else np.zeros((1, 1, 1, 1)))
        dummy = np.zeros(1, dtype=np.int64)
        codeV = self._side(vbasis).codes
        codeU = self._side(ubasis).codes if self._bilinear else dummy
        qd = (np.ascontiguousarray(np.concatenate(qdata, axis=-1))
              if qdata else np.zeros(1))
        P = np.array(params if params else [0.], dtype=np.float64)
        # data_as keeps a reference: invDF may be a temporary array
        vp = ctypes.c_void_p
        return lib, (ubasis.nelems, invDF.ctypes.data_as(vp), _strides(invDF),
                     dx.ctypes.data_as(vp), _strides(dx), qd, P,
                     coef, ubasis.N, idofs, gp, gp.shape[1], gconn,
                     DF.ctypes.data_as(vp), _strides(DF), codeU, codeV)

    def _c_assemble(self, ubasis, vbasis, kwargs):
        """Fill the COO data array of scikit-fem."""
        lib, args = self._prepare(ubasis, vbasis, kwargs)
        nu = ubasis.Nbfun if self._bilinear else 1
        data = np.empty(nu * vbasis.Nbfun * ubasis.nelems, dtype=np.float64)
        lib.assemble_coo(*args, data, self._nthreads)
        return data

    def _fallback(self, err):
        warnings.warn("{} falls back to Python assembly: {}"
                      .format(type(self).__name__, err))


_PARTITION_SOURCE = """
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

/* Group elements so that concurrently running threads write to disjoint
   rows.  Phase 0: compact blocks by recursive coordinate bisection, minus
   the interface elements sharing rows with other blocks.  Phases 1..: the
   interface elements colored greedily (no two elements of a color share a
   row), each color split between the threads. */

static void select_nth(int64_t *idx, int64_t n, int64_t k, const double *c)
{
  /* reorder idx so that c[idx[0..k-1]] <= c[idx[k]] <= c[idx[k+1..]] */
  int64_t lo = 0, hi = n - 1;
  while (lo < hi) {
    const int64_t m = lo + (hi - lo) / 2;
    /* median of three as the pivot; deterministic */
    double a = c[idx[lo]], b = c[idx[m]], d = c[idx[hi]];
    double pivot = a < b ? (b < d ? b : (a < d ? d : a))
                         : (a < d ? a : (b < d ? d : b));
    int64_t i = lo, j = hi;
    while (i <= j) {
      while (c[idx[i]] < pivot) i++;
      while (c[idx[j]] > pivot) j--;
      if (i <= j) {
        const int64_t t = idx[i]; idx[i] = idx[j]; idx[j] = t;
        i++; j--;
      }
    }
    if (k <= j) hi = j;
    else if (k >= i) lo = i;
    else return;
  }
}

static void bisect(int64_t *idx, int64_t n, int64_t nparts, int64_t first,
                   const double *mid, int64_t gdim, int64_t nt,
                   int64_t *block)
{
  if (nparts == 1) {
    for (int64_t e = 0; e < n; e++) block[idx[e]] = first;
    return;
  }
  /* cut along the longest extent */
  int64_t axis = 0;
  double best = -1.;
  for (int64_t d = 0; d < gdim; d++) {
    double lo = mid[d * nt + idx[0]], hi = lo;
    for (int64_t e = 1; e < n; e++) {
      const double x = mid[d * nt + idx[e]];
      if (x < lo) lo = x;
      if (x > hi) hi = x;
    }
    if (hi - lo > best) { best = hi - lo; axis = d; }
  }
  const int64_t half = nparts / 2;
  const int64_t cut = n * half / nparts;
  if (cut > 0 && cut < n) select_nth(idx, n, cut, mid + axis * nt);
  bisect(idx, cut, half, first, mid, gdim, nt, block);
  bisect(idx + cut, n - cut, nparts - half, first + half, mid, gdim, nt,
         block);
}

/* returns the number of groups, or -1 if 'maxgroups' is too small or
   memory runs out */
int64_t partition(int64_t nt, int64_t nbv, const int64_t *vdofs, int64_t N,
                  int64_t gdim, int64_t nverts, int64_t npts,
                  const double *p,
                  const int64_t *t, int64_t nthreads, int64_t minchunk,
                  int64_t *perm, int64_t *parts, int64_t *phases,
                  int64_t maxgroups, int64_t *nphases)
{
  int64_t ret = -1;
  double *mid = malloc(sizeof(double) * gdim * nt);
  int64_t *idx = malloc(sizeof(int64_t) * nt);
  int64_t *block = malloc(sizeof(int64_t) * nt);
  int64_t *color = malloc(sizeof(int64_t) * nt);
  int64_t *lmin = malloc(sizeof(int64_t) * N);
  int64_t *lmax = malloc(sizeof(int64_t) * N);
  int64_t *r2e_ptr = calloc(N + 1, sizeof(int64_t));
  int64_t *r2e = NULL, *mark = NULL, *count = NULL;
  if (!mid || !idx || !block || !color || !lmin || !lmax || !r2e_ptr)
    goto done;

  /* element midpoints */
  for (int64_t e = 0; e < nt; e++)
    for (int64_t d = 0; d < gdim; d++) {
      double sum = 0.;
      for (int64_t n = 0; n < nverts; n++)
        sum += p[d * npts + t[n * nt + e]];
      mid[d * nt + e] = sum / nverts;
    }

  /* phase 0: blocks by recursive coordinate bisection */
  for (int64_t e = 0; e < nt; e++) idx[e] = e;
  bisect(idx, nt, nthreads, 0, mid, gdim, nt, block);

  /* interface elements have a row shared with another block */
  for (int64_t r = 0; r < N; r++) { lmin[r] = nthreads; lmax[r] = -1; }
  for (int64_t e = 0; e < nt; e++)
    for (int64_t i = 0; i < nbv; i++) {
      const int64_t r = vdofs[i * nt + e];
      if (block[e] < lmin[r]) lmin[r] = block[e];
      if (block[e] > lmax[r]) lmax[r] = block[e];
    }
  int64_t ninterface = 0;
  for (int64_t e = 0; e < nt; e++) {
    color[e] = -1;
    for (int64_t i = 0; i < nbv; i++) {
      const int64_t r = vdofs[i * nt + e];
      if (lmin[r] != lmax[r]) { color[e] = 0; ninterface++; break; }
    }
  }

  /* row -> interface element map */
  for (int64_t e = 0; e < nt; e++)
    if (color[e] >= 0)
      for (int64_t i = 0; i < nbv; i++) r2e_ptr[vdofs[i * nt + e] + 1]++;
  for (int64_t r = 0; r < N; r++) r2e_ptr[r + 1] += r2e_ptr[r];
  r2e = malloc(sizeof(int64_t) * (r2e_ptr[N] > 0 ? r2e_ptr[N] : 1));
  if (!r2e) goto done;
  memcpy(lmin, r2e_ptr, sizeof(int64_t) * N);  /* reuse as positions */
  for (int64_t e = 0; e < nt; e++)
    if (color[e] >= 0)
      for (int64_t i = 0; i < nbv; i++) r2e[lmin[vdofs[i * nt + e]]++] = e;

  /* greedy coloring of interface elements in element order */
  int64_t ncolors = 0, cap = 64;
  mark = malloc(sizeof(int64_t) * cap);
  if (!mark) goto done;
  for (int64_t c = 0; c < cap; c++) mark[c] = -1;
  for (int64_t e = 0; e < nt; e++) color[e] = color[e] >= 0 ? -2 : -1;
  for (int64_t e = 0; e < nt; e++) {
    if (color[e] != -2) continue;
    for (int64_t i = 0; i < nbv; i++) {
      const int64_t r = vdofs[i * nt + e];
      for (int64_t q = r2e_ptr[r]; q < r2e_ptr[r + 1]; q++)
        if (color[r2e[q]] >= 0) mark[color[r2e[q]]] = e;
    }
    int64_t c = 0;
    while (c < cap && mark[c] == e) c++;
    if (c == cap) {
      int64_t *grown = realloc(mark, sizeof(int64_t) * 2 * cap);
      if (!grown) goto done;
      mark = grown;
      for (int64_t k = cap; k < 2 * cap; k++) mark[k] = -1;
      cap *= 2;
    }
    color[e] = c;
    if (c + 1 > ncolors) ncolors = c + 1;
  }

  /* groups: phase 0 has one group per block, phase 1 + c the elements of
     color c split in chunks; elements keep their order within groups */
  const int64_t nkeys = nthreads + ncolors;
  count = calloc(2 * nkeys + 1, sizeof(int64_t));
  if (!count) goto done;
  int64_t *start = count + nkeys;
  for (int64_t e = 0; e < nt; e++)
    count[color[e] < 0 ? block[e] : nthreads + color[e]]++;
  start[0] = 0;
  for (int64_t key = 0; key < nkeys; key++)
    start[key + 1] = start[key] + count[key];
  for (int64_t e = 0; e < nt; e++)
    perm[start[color[e] < 0 ? block[e] : nthreads + color[e]]++] = e;
  /* start[key] now points to the end of key; shift back */
  for (int64_t key = nkeys; key > 0; key--) start[key] = start[key - 1];
  start[0] = 0;

  int64_t ngroups = 0, nph = 0;
  phases[nph++] = 0;
  for (int64_t key = 0; key < nkeys; key++) {
    int64_t nchunks = 1;
    if (key >= nthreads) {
      nchunks = count[key] / minchunk;
      if (nchunks > nthreads) nchunks = nthreads;
      if (nchunks < 1) nchunks = 1;
    }
    if (ngroups + nchunks > maxgroups) goto done;
    for (int64_t g = 0; g < nchunks; g++)
      parts[ngroups++] = start[key] + count[key] * g / nchunks;
    if (key >= nthreads - 1) phases[nph++] = ngroups;
  }
  parts[ngroups] = nt;
  *nphases = nph - 1;
  ret = ngroups;

done:
  free(mid); free(idx); free(block); free(color); free(lmin); free(lmax);
  free(r2e_ptr); free(r2e); free(mark); free(count);
  return ret;
}
"""


def _partition_lib() -> ctypes.CDLL:
    lib = compile_source(_PARTITION_SOURCE)
    iptr = np.ctypeslib.ndpointer(np.int64, flags='C_CONTIGUOUS')
    dptr = np.ctypeslib.ndpointer(np.float64, flags='C_CONTIGUOUS')
    i64 = ctypes.c_int64
    lib.partition.argtypes = (i64, i64, iptr, i64, i64, i64, i64, dptr,
                              iptr, i64, i64, iptr, iptr, iptr, i64,
                              ctypes.POINTER(i64))
    lib.partition.restype = i64
    return lib


def _partition(basis, nthreads, minchunk=64):
    """Group elements so that concurrent threads write to disjoint rows.

    First phase: the elements are split into ``nthreads`` compact blocks by
    recursive coordinate bisection, and each thread assembles the elements
    of its block that do not share rows (test function DOFs) with other
    blocks.  Following phases: the remaining interface elements are colored
    greedily so that no two elements of a color share a row, and each color
    is divided between the threads.  Phases run one after another.  See
    ``_PARTITION_SOURCE``.

    Returns
    -------
    perm
        Elements ordered by group.
    parts
        Group ``g`` is ``perm[parts[g]:parts[g + 1]]``.
    phases
        Phase ``f`` consists of groups ``phases[f]:phases[f + 1]``; they
        are assembled in parallel.

    """
    vdofs = _dofs64(basis)
    nbv, nt = vdofs.shape
    if nthreads <= 1 or nt < 8 * nthreads:
        return (np.arange(nt, dtype=np.int64),
                np.array([0, nt], dtype=np.int64),
                np.array([0, 1], dtype=np.int64))
    mesh = basis.mesh
    t = mesh.t if basis.tind is None else mesh.t[:, basis.tind]
    t = np.ascontiguousarray(t, dtype=np.int64)
    p = np.ascontiguousarray(mesh.p, dtype=np.float64)
    lib = _partition_lib()
    maxgroups = 256 * nthreads
    while True:
        perm = np.empty(nt, dtype=np.int64)
        parts = np.empty(maxgroups + 1, dtype=np.int64)
        phases = np.empty(maxgroups + 1, dtype=np.int64)
        nphases = ctypes.c_int64()
        ngroups = lib.partition(nt, nbv, vdofs, basis.N, p.shape[0],
                                t.shape[0], p.shape[1], p, t, nthreads,
                                minchunk, perm, parts, phases, maxgroups,
                                ctypes.byref(nphases))
        if ngroups >= 0:
            return (perm, parts[:ngroups + 1].copy(),
                    phases[:nphases.value + 1].copy())
        if maxgroups > nthreads * (nt + 1):
            raise MemoryError("Partitioning the elements failed.")
        maxgroups *= 8


class _Sparsity:
    """Cached CSR sparsity pattern and thread partition."""

    def __init__(self, ubasis, vbasis, nthreads):
        self.shape = (vbasis.N, ubasis.N)
        self.udofs = _dofs64(ubasis)
        self.vdofs = _dofs64(vbasis)
        self.indptr, self.indices = sparsity_pattern(
            self.udofs, self.vdofs, ubasis.N, vbasis.N, nthreads)
        self.perm, self.parts, self.phases = _partition(vbasis, nthreads)

    _digests = weakref.WeakKeyDictionary()

    @classmethod
    def _digest(cls, basis):
        """Hash of the DOF numbering; remembered per Basis object."""
        try:
            return cls._digests[basis]
        except (KeyError, TypeError):
            pass
        h = hashlib.sha256()
        h.update(np.ascontiguousarray(basis.element_dofs).data)
        h.update(np.array([basis.N, basis.Nbfun, basis.nelems]).data)
        digest = h.hexdigest()
        try:
            cls._digests[basis] = digest
        except TypeError:  # not weak referenceable
            pass
        return digest

    @classmethod
    def key(cls, ubasis, vbasis, nthreads):
        return cls._digest(ubasis), cls._digest(vbasis), nthreads


class CBilinearForm(_CFormMixin, BilinearForm):
    """A :class:`~skfem.assembly.BilinearForm` assembled by generated C.

    Use exactly as :class:`~skfem.assembly.BilinearForm`.  Integrands may use
    :mod:`skfem.helpers`, arithmetic, elementary functions (``np.sqrt``,
    ``np.exp``, ...), scalars and fields given through ``w``.

    Parameters
    ----------
    nthreads
        Number of POSIX threads; the default ``0`` means all cores.
    cache_sparsity
        If ``True``, the CSR sparsity pattern is computed on the first call
        to :meth:`assemble` and cached (per pair of bases), and the local
        matrices are added directly into the CSR data array.  The resulting
        matrices then have identical sparsity patterns, including possible
        explicit zeros.

    """

    _bilinear = True

    def __init__(self, form=None, dtype=np.float64, nthreads=0,
                 cache_sparsity=False, **params):
        # stored in params so that the decorator syntax passes it on
        super().__init__(form, dtype=dtype, nthreads=nthreads,
                         cache_sparsity=cache_sparsity, **params)
        self.cache_sparsity = cache_sparsity
        self._sparsity = {}

    def sparsity(self, ubasis, vbasis=None) -> _Sparsity:
        """Return the cached sparsity pattern; compute if not cached."""
        vbasis = ubasis if vbasis is None else vbasis
        key = _Sparsity.key(ubasis, vbasis, self._nthreads)
        if key not in self._sparsity:
            self._sparsity[key] = _Sparsity(ubasis, vbasis, self._nthreads)
        return self._sparsity[key]

    def assemble(self, ubasis, vbasis=None, **kwargs):
        if (not self.cache_sparsity
                or self._distributed_dofs(ubasis, vbasis) is not None):
            return super().assemble(ubasis, vbasis, **kwargs)
        from scipy.sparse import csr_matrix
        vbasis = ubasis if vbasis is None else vbasis
        try:
            self._check_support(ubasis, vbasis)
            # pattern first; its index dtype is used in the generated code
            S = self.sparsity(ubasis, vbasis)
            lib, args = self._prepare(ubasis, vbasis, dict(kwargs),
                                      S.indices.dtype)
        except _Unsupported as err:
            self._fallback(err)
            return super().assemble(ubasis, vbasis, **kwargs)
        data = np.zeros(len(S.indices), dtype=np.float64)
        lib.assemble_csr(*args, S.indptr, S.indices, S.udofs, S.vdofs,
                         S.perm, S.parts, S.phases, len(S.phases) - 1, data)
        return csr_matrix((data, S.indices, S.indptr), shape=S.shape)

    def _assemble(self, ubasis, vbasis=None, **kwargs):
        vbasis = ubasis if vbasis is None else vbasis
        try:
            data = self._c_assemble(ubasis, vbasis, dict(kwargs))
        except _Unsupported as err:
            self._fallback(err)
            return super()._assemble(ubasis, vbasis, **kwargs)
        rowcol = np.empty((2, ubasis.Nbfun, vbasis.Nbfun, ubasis.nelems),
                          dtype=np.int32)
        rowcol[0] = vbasis.element_dofs[None, :, :]
        rowcol[1] = ubasis.element_dofs[:, None, :]
        return (
            rowcol.reshape(2, -1),
            data,
            (vbasis.N, ubasis.N),
            (vbasis.Nbfun, ubasis.Nbfun),
        )


class CLinearForm(_CFormMixin, LinearForm):
    """A :class:`~skfem.assembly.LinearForm` assembled by generated C."""

    _bilinear = False

    def _assemble(self, ubasis, vbasis=None, **kwargs):
        assert vbasis is None
        try:
            data = self._c_assemble(ubasis, ubasis, dict(kwargs))
        except _Unsupported as err:
            self._fallback(err)
            return super()._assemble(ubasis, **kwargs)
        rows = np.ascontiguousarray(ubasis.element_dofs, dtype=np.int32)
        return (rows.reshape(1, -1), data, (ubasis.N,), (ubasis.Nbfun,))
