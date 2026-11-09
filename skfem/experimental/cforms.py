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

2. *Code generation.*  The reference basis functions are tabulated at the
   quadrature points in Python (via ``Element.lbasis``) and embedded into the
   C source as constant arrays.  The element loop transforms them using
   ``invDF`` and ``dx`` that are taken from the Basis object, so any mapping
   (affine, isoparametric) is supported without writing C for it.  DOF
   vectors given to ``assemble`` and the global coordinates ``w.x`` are
   interpolated inside the kernel.

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

Unsupported cases (e.g. facet bases, H(div)/H(curl) elements, complex
dtype, no C compiler as in Pyodide) fall back to the Python implementation
with a warning.

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
from skfem.element import (DiscreteField, ElementDG, ElementH1,
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


def _deps(name: str) -> frozenset:
    """Loop indices a symbol depends on: trial 'j', test 'i' or none."""
    if name.startswith(('U[', 'DU[')):
        return frozenset('j')
    if name.startswith(('V[', 'DV[')):
        return frozenset('i')
    return frozenset()


class CExpr:
    """A scalar C expression; supports the arithmetic used in forms.

    Expressions are nodes of a graph: a symbol (a C variable), or an
    operation ``op`` applied to ``args`` (CExprs or numbers).  Identical
    subexpressions are the same node, which is used for hoisting.

    """

    __slots__ = ('op', 'args', 'deps', 'id')

    @staticmethod
    def symbol(name: str) -> 'CExpr':
        def make():
            node = CExpr()
            node.op, node.args, node.deps = 'sym', (name,), _deps(name)
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
    del _fun


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


def _emit(root, bilinear: bool):
    """Generate the loops over basis functions with hoisted temporaries.

    Every operation is computed once at the outermost loop level allowed by
    its dependencies: per quadrature point (only 'w'), per test function
    'i', per trial function 'j' (precomputed to arrays over 'j' before the
    loops), or in the innermost loop (both 'i' and 'j').

    """
    if not isinstance(root, CExpr):
        expr = repr(_const(root))
        nodes = []
    else:
        # operations reachable from the root, in post-order
        nodes, seen, stack = [], set(), [(root, False)]
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
    code = {'q': [], 'i': [], 'j': [], 'ij': []}
    jdecl = []

    def ref(node):
        return names[node.id] if node.op != 'sym' else node.args[0]

    for n, node in enumerate(nodes):
        level = ''.join(sorted(node.deps)) or 'q'
        src = _render(node, ref)
        if level == 'j':
            names[node.id] = "tj{}[j]".format(n)
            jdecl.append("tj{}[NU]".format(n))
            code['j'].append("      tj{}[j] = {};".format(n, src))
        else:
            names[node.id] = "t{}{}".format(level, n)
            indent = {'q': 4, 'i': 6, 'ij': 8}[level]
            code[level].append("{}const double t{}{} = {};"
                               .format(' ' * indent, level, n, src))
    if isinstance(root, CExpr):
        expr = ref(root)

    if bilinear:
        jcode = ""
        if code['j']:
            jcode = ("    double " + ", ".join(jdecl) + ";\n"
                     "    for (int64_t j = 0; j < NU; j++) {\n"
                     + "\n".join(code['j']) + "\n    }")
        return _BILINEAR_LOOPS.format(
            qcode="\n".join(code['q']),
            jcode=jcode,
            icode="\n".join(code['i']),
            ijcode="\n".join(code['ij']),
            expr=expr,
        )
    assert not code['j'] and not code['ij']
    return _LINEAR_LOOPS.format(
        qcode="\n".join(code['q']),
        icode="\n".join(code['i']),
        expr=expr,
    )


def _symbols(fmt: str, shape: Tuple[int, ...], offset: int = 0) -> ndarray:
    """Object array of CExpr's named by ``fmt.format(offset + flat_index)``.

    Two trailing axes of length one are appended so that the arrays have the
    same rank as the numerical arrays (..., nelems, nqp) seen by forms;
    shape-based helpers such as ``inner`` or ``identity`` then work as is.

    """
    out = np.empty(shape, dtype=object)
    for n, ix in enumerate(np.ndindex(*shape)):
        out[ix] = CExpr.symbol(fmt.format(offset + n))
    return out.reshape(shape + (1, 1))


def _symbolic_field(name: str, index: str, ncomp: int,
                    gdim: int) -> DiscreteField:
    """Symbolic basis function: C arrays ``name[j][c]``, ``Dname[j][n]``.

    Here ``j`` is the local basis function index (a loop variable in C
    called ``index``), ``c`` the component and ``g`` the direction of the
    derivative; ``n = c * GDIM + g``.

    """
    vshape = () if ncomp == 1 else (ncomp,)
    pre = "{}[{}]".format(name, index)
    dpre = "D{}[{}]".format(name, index)
    return DiscreteField(
        value=_symbols(pre + "[{}]", vshape),
        grad=_symbols(dpre + "[{}]", vshape + (gdim,)),
    )


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
# Element tabulation
# ---------------------------------------------------------------------------

class _Unsupported(Exception):
    pass


def _tabulate(elem, X: ndarray, nbfun: int):
    """Tabulate reference basis functions at quadrature points.

    Returns
    -------
    phi
        Values of the scalar reference functions, shape (nscalar, nqp).
    dphi
        Reference gradients, shape (nscalar, ldim, nqp).
    base, comp
        For each local basis function, the index of the scalar reference
        function and the component it is placed to.
    ncomp
        Number of components.

    """
    if isinstance(elem, ElementVector):
        phi, dphi, base, comp, ncomp = _tabulate(elem.elem, X,
                                                 nbfun // elem.dim)
        if ncomp != 1:
            raise _Unsupported("Nested ElementVector")
        n = np.arange(nbfun)
        return phi, dphi, base[n // elem.dim], n % elem.dim, elem.dim
    if isinstance(elem, ElementDG):
        return _tabulate(elem.elem, X, nbfun)
    if not isinstance(elem, ElementH1):
        raise _Unsupported(type(elem).__name__)
    if type(elem).gbasis is not ElementH1.gbasis:
        raise _Unsupported("Custom gbasis in " + type(elem).__name__)
    if (elem.dim == 2
            and elem.facet_dofs >= 2
            and elem.reversed_facet_dof(0) is not None):
        # basis depends on facet orientation, i.e. differs per element
        raise _Unsupported("Orientation-dependent " + type(elem).__name__)
    nqp = X.shape[-1]
    phi = np.empty((nbfun, nqp))
    dphi = np.empty((nbfun, X.shape[0], nqp))
    for i in range(nbfun):
        p, dp = elem.lbasis(X, i)
        phi[i] = np.broadcast_to(p, (nqp,))
        dphi[i] = np.broadcast_to(dp, (X.shape[0], nqp))
    n = np.arange(nbfun)
    return phi, dphi, n, np.zeros_like(n), 1


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------

def _carray(a: ndarray) -> str:
    """Nested C initializer list."""
    if a.ndim == 0:
        return repr(a.item())
    return "{" + ", ".join(_carray(x) for x in a) + "}"


_BASIS_TEMPLATE = """
static const double PHI_{N}[{nscalar}][NQ] = {phi};
static const double DPHI_{N}[{nscalar}][LDIM][NQ] = {dphi};
static const int64_t BASE_{N}[N{N}] = {base};
static const int64_t COMP_{N}[N{N}] = {comp};
"""


_EVAL_TEMPLATE = """
    /* global basis functions {N} at quadrature point q */
    memset({N}, 0, sizeof({N}));
    memset(D{N}, 0, sizeof(D{N}));
    for (int64_t j = 0; j < N{N}; j++) {{
      const int64_t b = BASE_{N}[j], c = COMP_{N}[j];
      {N}[j][c] = PHI_{N}[b][q];
      for (int g = 0; g < GDIM; g++) {{
        double s = 0.;
        for (int a = 0; a < LDIM; a++)
          s += J[a * GDIM + g] * DPHI_{N}[b][a][q];
        D{N}[j][c * GDIM + g] = s;
      }}
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
#define NCV {ncv}
#define NU {nu}
#define NCU {ncu}
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
  double V[NV][NCV], DV[NV][NCV * GDIM];
{udecl}
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
                  double *data,
                  int64_t nthreads)
{{
  struct input in = {{.nt = nt, .invDF = invDF, .js = js, .dx = dx,
                      .dxs = dxs, .qd = qd, .P = P, .coef = coef,
                      .coef_n = coef_n, .idofs = idofs, .gp = gp,
                      .gnp = gnp, .gconn = gconn}};
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
                      .gnp = gnp, .gconn = gconn, .indptr = indptr,
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


_GATHER_TEMPLATE = """
  /* local coefficients of the DOF vector '{name}' of 'w' */
  double C{f}[N{B}];
  for (int64_t j = 0; j < N{B}; j++)
    C{f}[j] = in->coef[{f} * in->coef_n + in->idofs[j * in->nt + k]];"""


_INTERP_TEMPLATE = """
    /* '{name}' of 'w' interpolated at quadrature point q */
    double F{f}[NC{B}], DF{f}[NC{B} * GDIM];
    memset(F{f}, 0, sizeof(F{f}));
    memset(DF{f}, 0, sizeof(DF{f}));
    for (int64_t j = 0; j < N{B}; j++) {{
      for (int n = 0; n < NC{B}; n++)
        F{f}[n] += C{f}[j] * {B}[j][n];
      for (int n = 0; n < NC{B} * GDIM; n++)
        DF{f}[n] += C{f}[j] * D{B}[j][n];
    }}"""


_GEOMETRY_TEMPLATE = """
    /* global coordinates 'x' at quadrature point q */
    double XQ[GDIM];
    for (int g = 0; g < GDIM; g++) XQ[g] = 0.;
    for (int64_t n = 0; n < NG; n++) {{
      const int64_t node = in->gconn[n * in->nt + k];
      for (int g = 0; g < GDIM; g++)
        XQ[g] += in->gp[g * in->gnp + node] * GPHI[n][q];
    }}"""


_BILINEAR_LOOPS = """
{qcode}
{jcode}
    for (int64_t i = 0; i < NV; i++) {{
{icode}
      for (int64_t j = 0; j < NU; j++) {{
{ijcode}
        A[j * NV + i] += {expr} * w_dx;
      }}
    }}"""


_LINEAR_LOOPS = """
{qcode}
    for (int64_t i = 0; i < NV; i++) {{
{icode}
      A[i] += {expr} * w_dx;
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
                                 dptr, i64, iptr, dptr, i64, iptr, dptr,
                                 i64)
    lib.assemble_coo.restype = None
    lib.assemble_csr.argtypes = (i64, vptr, iptr, vptr, iptr, dptr, dptr,
                                 dptr, i64, iptr, dptr, i64, iptr, xptr,
                                 xptr, iptr, iptr, iptr, iptr, iptr, i64,
                                 dptr)
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

    def _tables(self, name, basis):
        phi, dphi, base, comp, ncomp = _tabulate(basis.elem, basis.X,
                                                 basis.Nbfun)
        src = _BASIS_TEMPLATE.format(
            N=name,
            nscalar=phi.shape[0],
            phi=_carray(phi),
            dphi=_carray(dphi),
            base=_carray(base),
            comp=_carray(comp),
        )
        return src, ncomp

    @staticmethod
    def _build_w(wdict, coefs=(), cshape=(1, 1), used=None):
        """Symbolic 'w'.

        DiscreteFields in ``used`` are packed into ``Q``; DOF vectors with
        names in ``coefs`` are interpolated in C to ``F<n>`` and ``DF<n>``,
        where ``cshape`` is (number of components, gdim).

        """
        w = _TracingParams()
        params, qdata, offset = [], [], 0
        ncomp, gdim = cshape
        vshape = () if ncomp == 1 else (ncomp,)
        interpolated = []
        for key in coefs:
            if used is not None and key not in used:
                continue
            f = len(interpolated)
            interpolated.append(key)
            dict.__setitem__(w, key, DiscreteField(
                value=_symbols("F{}[{{}}]".format(f), vshape),
                grad=_symbols("DF{}[{{}}]".format(f), vshape + (gdim,)),
            ))
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
            elif isinstance(val, DiscreteField):
                if used is not None and key not in used:
                    continue
                # fields are passed as values at the quadrature points
                syms = {}
                for attr in ('value', 'grad'):
                    arr = np.asarray(val) if attr == 'value' else val.grad
                    if arr is None:
                        syms[attr] = None
                        continue
                    shape = np.shape(arr)[:-2]
                    syms[attr] = _symbols("Q[{}]", shape, offset)
                    offset += int(np.prod(shape))
                    if used is not None:
                        arr = np.asarray(arr, dtype=np.float64)
                        nt, nq = arr.shape[-2:]
                        qdata.append(np.moveaxis(arr.reshape(-1, nt, nq),
                                                 0, -1))
                dict.__setitem__(w, key, DiscreteField(
                    value=syms['value'],
                    grad=syms['grad'],
                ))
            else:
                # e.g. 'idx' of asm(); keep as is
                dict.__setitem__(w, key, val)
        return w, params, qdata, offset, interpolated, uses_x

    def _trace(self, args, wdict, coefs=(), cshape=(1, 1)):
        """Call the form with symbolic arguments.

        The first call finds out which entries of 'w' are used, the second
        builds the final expression; only the used fields are packed.

        """
        try:
            w = self._build_w(wdict, coefs, cshape)[0]
            self.form(*args, w)
            w, params, qdata, nqd, interp, uses_x = self._build_w(
                wdict, coefs, cshape, used=w.accessed)
            expr = _simplify_result(self.form(*args, w))
        except (TypeError, NotImplementedError) as err:
            # e.g. branching on values or NumPy functions without C version
            raise _Unsupported("Tracing the form failed: {}".format(err))
        return expr, params, qdata, nqd, interp, uses_x

    def _generate(self, basis, vbasis, wdict, idx_dtype=np.int32,
                  coefs=(), geom=None):
        global _graph
        _graph = _Graph()  # fresh expression graph for this trace
        ttab, ncv = self._tables('V', vbasis)
        # not probed from invDF: isoparametric mappings cache only the
        # latest invDF, i.e., the one of the basis that is reused below
        gdim = basis.mesh.p.shape[0]
        # trial function is indexed by 'j' and test function by 'i' in C
        if self._bilinear:
            utab, ncu = self._tables('U', basis)
            args = (_symbolic_field('U', 'j', ncu, gdim),
                    _symbolic_field('V', 'i', ncv, gdim))
        else:
            utab, ncu = "", 1
            args = (_symbolic_field('V', 'i', ncv, gdim),)

        # DOF vectors are interpolated using the trial basis (as in
        # scikit-fem), which is the test basis for linear forms
        B, ncb = ('U', ncu) if self._bilinear else ('V', ncv)
        expr, params, qdata, nqd, interp, uses_x = self._trace(
            args, wdict, coefs, (ncb, gdim))
        gather = "".join(_GATHER_TEMPLATE.format(name=name, f=f, B=B)
                         for f, name in enumerate(interp))
        interp_src = "".join(_INTERP_TEMPLATE.format(name=name, f=f, B=B)
                             for f, name in enumerate(interp))
        gtab = ""
        if uses_x:
            gtab = ("\n#define NG {}\nstatic const double GPHI[NG][NQ] = {};\n"
                    .format(geom[0].shape[0], _carray(geom[0])))
            interp_src = _GEOMETRY_TEMPLATE.format() + interp_src

        if self._bilinear:
            udecl = "  double U[NU][NCU], DU[NU][NCU * GDIM];"
            evals = (_EVAL_TEMPLATE.format(N='U')
                     + _EVAL_TEMPLATE.format(N='V'))
            loops = _emit(expr, bilinear=True)
        else:
            udecl = ""
            evals = _EVAL_TEMPLATE.format(N='V')
            loops = _emit(expr, bilinear=False)

        source = _KERNEL_TEMPLATE.format(
            nqp=basis.X.shape[-1],
            ldim=basis.X.shape[0],
            gdim=gdim,
            nv=vbasis.Nbfun,
            ncv=ncv,
            nu=basis.Nbfun if self._bilinear else 1,
            ncu=ncu,
            nqd=max(nqd, 1),
            idx_t={np.int32: 'int32_t',
                   np.int64: 'int64_t'}[np.dtype(idx_dtype).type],
            tables=utab + ttab + gtab,
            udecl=udecl,
            gather=gather,
            eval=evals,
            interp=interp_src,
            loops=loops,
        )
        return source, params, qdata, interp, uses_x

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
        source, params, qdata, interp, uses_x = self._generate(
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
        qd = (np.ascontiguousarray(np.concatenate(qdata, axis=-1))
              if qdata else np.zeros(1))
        P = np.array(params if params else [0.], dtype=np.float64)
        # data_as keeps a reference: invDF may be a temporary array
        vp = ctypes.c_void_p
        return lib, (ubasis.nelems, invDF.ctypes.data_as(vp), _strides(invDF),
                     dx.ctypes.data_as(vp), _strides(dx), qd, P,
                     coef, ubasis.N, idofs, gp, gp.shape[1], gconn)

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
        if not self.cache_sparsity:
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
