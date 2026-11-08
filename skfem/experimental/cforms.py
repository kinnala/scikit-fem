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

2. *Code generation.*  The reference basis functions are tabulated at the
   quadrature points in Python (via ``Element.lbasis``) and embedded into the
   C source as constant arrays.  The element loop transforms them using
   ``invDF`` and ``dx`` that are taken from the Basis object, so any mapping
   (affine, isoparametric) is supported without writing C for it.

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
dtype) fall back to the Python implementation with a warning.

"""
import ctypes
import hashlib
import numbers
import os
import subprocess
import tempfile
import warnings
import weakref
from typing import Dict, Tuple

import numpy as np
from numpy import ndarray

from skfem.assembly.basis import CellBasis
from skfem.assembly.form import BilinearForm, LinearForm
from skfem.assembly.form.form import FormExtraParams
from skfem.element import (DiscreteField, ElementDG, ElementH1,
                           ElementVector)


# ---------------------------------------------------------------------------
# Symbolic scalars
# ---------------------------------------------------------------------------

def _c(x) -> str:
    """Convert a CExpr or a number into C source."""
    if isinstance(x, CExpr):
        return x.src
    if isinstance(x, numbers.Number):
        if isinstance(x, complex):
            raise NotImplementedError("Complex numbers are not supported.")
        return repr(float(x))
    raise TypeError("Cannot convert '{}' to C.".format(type(x)))


def _is(x, value) -> bool:
    return isinstance(x, numbers.Number) and x == value


class CExpr:
    """A scalar C expression; supports the arithmetic used in forms."""

    __slots__ = ('src',)

    def __init__(self, src: str):
        self.src = src

    def __repr__(self):
        return self.src

    # arithmetic, with trivial simplifications so that the zero blocks of
    # e.g. vectorial fields do not end up in the generated source
    def __add__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return self
        return CExpr("({} + {})".format(self.src, _c(other)))

    def __radd__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return self
        return CExpr("({} + {})".format(_c(other), self.src))

    def __sub__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return self
        return CExpr("({} - {})".format(self.src, _c(other)))

    def __rsub__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return -self
        return CExpr("({} - {})".format(_c(other), self.src))

    def __mul__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return 0.
        if _is(other, 1):
            return self
        return CExpr("({} * {})".format(self.src, _c(other)))

    def __rmul__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return 0.
        if _is(other, 1):
            return self
        return CExpr("({} * {})".format(_c(other), self.src))

    def __truediv__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        return CExpr("({} / {})".format(self.src, _c(other)))

    def __rtruediv__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if _is(other, 0):
            return 0.
        return CExpr("({} / {})".format(_c(other), self.src))

    def __pow__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        if isinstance(other, numbers.Integral) and 0 < other <= 4:
            return CExpr("(" + " * ".join([self.src] * int(other)) + ")")
        return CExpr("pow({}, {})".format(self.src, _c(other)))

    def __rpow__(self, other):
        if not isinstance(other, (CExpr, numbers.Number)):
            return NotImplemented
        return CExpr("pow({}, {})".format(_c(other), self.src))

    def __neg__(self):
        return CExpr("(-{})".format(self.src))

    def __pos__(self):
        return self

    def __abs__(self):
        return CExpr("fabs({})".format(self.src))

    def __bool__(self):
        raise TypeError("Symbolic expressions have no truth value; "
                        "branching on values is not supported in C forms.")

    # NumPy calls these methods for ufuncs on object arrays, e.g. np.sqrt
    def _fun(name):
        def f(self):
            return CExpr("{}({})".format(name, self.src))
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


def _symbols(fmt: str, shape: Tuple[int, ...], offset: int = 0) -> ndarray:
    """Object array of CExpr's named by ``fmt.format(offset + flat_index)``.

    Two trailing axes of length one are appended so that the arrays have the
    same rank as the numerical arrays (..., nelems, nqp) seen by forms;
    shape-based helpers such as ``inner`` or ``identity`` then work as is.

    """
    out = np.empty(shape, dtype=object)
    for n, ix in enumerate(np.ndindex(*shape)):
        out[ix] = CExpr(fmt.format(offset + n))
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
    """Turn the traced integrand into a C expression string."""
    out = np.asarray(out, dtype=object)
    if out.size != 1:
        raise ValueError("The form must return a scalar, got an array of "
                         "shape {}.".format(out.shape[:-2]))
    return _c(out.reshape(-1)[0])


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
                  double *data,
                  int64_t nthreads)
{{
  struct input in = {{.nt = nt, .invDF = invDF, .js = js, .dx = dx,
                      .dxs = dxs, .qd = qd, .P = P, .coef = coef,
                      .coef_n = coef_n, .idofs = idofs}};
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
                      .coef_n = coef_n, .idofs = idofs, .indptr = indptr,
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


_BILINEAR_LOOPS = """
    for (int64_t i = 0; i < NV; i++) {{
      for (int64_t j = 0; j < NU; j++) {{
        A[j * NV + i] += ({expr}) * w_dx;
      }}
    }}"""


_LINEAR_LOOPS = """
    for (int64_t i = 0; i < NV; i++) {{
      A[i] += ({expr}) * w_dx;
    }}"""


def _cache_dir() -> str:
    path = os.environ.get(
        'SKFEM_CFORMS_CACHE',
        os.path.join(os.path.expanduser('~'), '.cache', 'skfem', 'cforms'),
    )
    os.makedirs(path, exist_ok=True)
    return path


_libraries: Dict[str, ctypes.CDLL] = {}


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
                                 dptr, i64, iptr, dptr, i64)
    lib.assemble_coo.restype = None
    lib.assemble_csr.argtypes = (i64, vptr, iptr, vptr, iptr, dptr, dptr,
                                 dptr, i64, iptr, xptr, xptr, iptr, iptr,
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


_dofs64_cache = weakref.WeakKeyDictionary()


def _dofs64(basis) -> ndarray:
    """``basis.element_dofs`` as contiguous int64; remembered per Basis."""
    try:
        return _dofs64_cache[basis]
    except (KeyError, TypeError):
        pass
    dofs = np.ascontiguousarray(basis.element_dofs, dtype=np.int64)
    try:
        _dofs64_cache[basis] = dofs
    except TypeError:
        pass
    return dofs


def _strides(arr) -> ndarray:
    """Strides in units of float64."""
    return np.array(arr.strides, dtype=np.int64) // 8


class _CFormMixin:
    """Shared logic of CBilinearForm and CLinearForm."""

    def _check_support(self, *bases):
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
        for key, val in wdict.items():
            if isinstance(val, numbers.Number):
                # runtime parameter: changing the value needs no recompile
                dict.__setitem__(w, key, CExpr("P[{}]".format(len(params))))
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
        return w, params, qdata, offset, interpolated

    def _trace(self, args, wdict, coefs=(), cshape=(1, 1)):
        """Call the form with symbolic arguments.

        The first call finds out which entries of 'w' are used, the second
        builds the final expression; only the used fields are packed.

        """
        try:
            w = self._build_w(wdict, coefs, cshape)[0]
            self.form(*args, w)
            w, params, qdata, nqd, interp = self._build_w(
                wdict, coefs, cshape, used=w.accessed)
            expr = _simplify_result(self.form(*args, w))
        except (TypeError, NotImplementedError) as err:
            # e.g. branching on values or NumPy functions without C version
            raise _Unsupported("Tracing the form failed: {}".format(err))
        return expr, params, qdata, nqd, interp

    def _generate(self, basis, vbasis, wdict, idx_dtype=np.int32,
                  coefs=()):
        ttab, ncv = self._tables('V', vbasis)
        gdim = basis.mapping.invDF(basis.X[:, :1],
                                   tind=np.array([0])).shape[1]
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
        expr, params, qdata, nqd, interp = self._trace(args, wdict, coefs,
                                                       (ncb, gdim))
        gather = "".join(_GATHER_TEMPLATE.format(name=name, f=f, B=B)
                         for f, name in enumerate(interp))
        interp_src = "".join(_INTERP_TEMPLATE.format(name=name, f=f, B=B)
                             for f, name in enumerate(interp))

        if self._bilinear:
            udecl = "  double U[NU][NCU], DU[NU][NCU * GDIM];"
            evals = (_EVAL_TEMPLATE.format(N='U')
                     + _EVAL_TEMPLATE.format(N='V'))
            loops = _BILINEAR_LOOPS.format(expr=expr)
        else:
            udecl = ""
            evals = _EVAL_TEMPLATE.format(N='V')
            loops = _LINEAR_LOOPS.format(expr=expr)

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
            tables=utab + ttab,
            udecl=udecl,
            gather=gather,
            eval=evals,
            interp=interp_src,
            loops=loops,
        )
        return source, params, qdata, interp

    def source(self, ubasis, vbasis=None, **kwargs) -> str:
        """Return the generated C source (for inspection)."""
        vbasis = ubasis if vbasis is None else vbasis
        kwargs = dict(kwargs)
        coefs = self._split_coefs(ubasis, kwargs)
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })
        return self._generate(ubasis, vbasis, wdict, coefs=coefs)[0]

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
        kwargs = dict(kwargs)
        coefs = self._split_coefs(ubasis, kwargs)
        wdict = FormExtraParams({
            **ubasis.default_parameters(),
            **self._normalize_asm_kwargs(kwargs, ubasis),
        })
        source, params, qdata, interp = self._generate(
            ubasis, vbasis, wdict, idx_dtype, coefs=list(coefs))
        lib = _kernel_lib(source)
        if interp:
            coef = np.ascontiguousarray(np.stack([coefs[k] for k in interp]),
                                        dtype=np.float64)
            idofs = _dofs64(ubasis)
        else:
            coef, idofs = np.zeros(1), np.zeros(1, dtype=np.int64)
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
                     coef, ubasis.N, idofs)

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


def _bisect(mid, idx, nparts):
    """Recursive coordinate bisection of the elements ``idx``."""
    if nparts == 1:
        return [idx]
    axis = np.argmax(np.ptp(mid[:, idx], axis=1))
    idx = idx[np.argsort(mid[axis, idx], kind='stable')]
    half = nparts // 2
    cut = len(idx) * half // nparts
    return (_bisect(mid, idx[:cut], half)
            + _bisect(mid, idx[cut:], nparts - half))


def _independent_sets(vdofs, idx, N):
    """Split elements ``idx`` to sets of elements with no common rows.

    Jones-Plassmann: in each round, the elements whose random priority is
    the largest among the remaining elements sharing any of its rows form
    an independent set.

    """
    prio = np.random.default_rng(0).permutation(len(idx)) + 1
    sets = []
    rem = np.arange(len(idx))
    while len(rem) > 0:
        dofs = vdofs[:, idx[rem]]
        rmax = np.zeros(N, dtype=np.int64)
        np.maximum.at(rmax, dofs.ravel(),
                      np.broadcast_to(prio[rem], dofs.shape).ravel())
        sel = (rmax[dofs] == prio[rem]).all(axis=0)
        sets.append(idx[rem[sel]])
        rem = rem[~sel]
    return sets


def _partition(basis, nthreads):
    """Group elements so that concurrent threads write to disjoint rows.

    First phase: the elements are split into ``nthreads`` compact blocks by
    recursive coordinate bisection, and each thread assembles the elements
    of its block that do not share rows (test function DOFs) with other
    blocks.  Following phases: the remaining interface elements are split
    into independent sets (no two elements share a row), and each set is
    divided between the threads.  Phases run one after another.

    This is the idea of femgen without the METIS dependency, and without
    the extra output buffer for the interface elements.

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
    vdofs = basis.element_dofs
    nt = vdofs.shape[1]
    if nthreads <= 1 or nt < 8 * nthreads:
        return (np.arange(nt, dtype=np.int64),
                np.array([0, nt], dtype=np.int64),
                np.array([0, 1], dtype=np.int64))
    mesh = basis.mesh
    t = mesh.t if basis.tind is None else mesh.t[:, basis.tind]
    mid = mesh.p[:, t].mean(axis=1)

    # first phase: interiors of compact blocks
    blocks = _bisect(mid, np.arange(nt), nthreads)
    label = np.empty(nt, dtype=np.int64)
    for n, block in enumerate(blocks):
        label[block] = n
    labels = np.broadcast_to(label, vdofs.shape).ravel()
    lmin = np.full(basis.N, nthreads, dtype=np.int64)
    lmax = np.full(basis.N, -1, dtype=np.int64)
    np.minimum.at(lmin, vdofs.ravel(), labels)
    np.maximum.at(lmax, vdofs.ravel(), labels)
    interface = (lmin != lmax)[vdofs].any(axis=0)
    groups = [block[~interface[block]] for block in blocks]
    phases = [0, len(groups)]

    # following phases: independent sets of interface elements
    for iset in _independent_sets(vdofs, np.nonzero(interface)[0], basis.N):
        nchunks = int(min(nthreads, max(1, len(iset) // 64)))
        groups += np.array_split(iset, nchunks)
        phases.append(len(groups))

    perm = np.concatenate(groups).astype(np.int64)
    parts = np.concatenate(([0], np.cumsum([len(g) for g in groups])))
    return perm, parts.astype(np.int64), np.array(phases, dtype=np.int64)


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
