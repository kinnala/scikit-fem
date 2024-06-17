import skfem
import io
import numpy as np

from typing import Optional
from glvis import glvis

from skfem.generic_utils import deprecated


MESH_TYPE_MAPPING = {
    skfem.MeshTet1: 4,
    skfem.MeshTri1: 2,
    skfem.MeshQuad1: 3,
    skfem.MeshLine1: 1,
}


BOUNDARY_TYPE_MAPPING = {
    4: 2,
    2: 1,
    3: 1,
    5: 3,
}


template = """solution

MFEM mesh v1.0

dimension
{}

elements
{}
{}

boundary
{}
{}

vertices
{}
{}
{}

FiniteElementSpace
FiniteElementCollection: {}
VDim: {}
Ordering: 0
{}

{}
"""


def _to_int_string(arr):
    s = io.BytesIO()
    np.savetxt(s, arr, delimiter=' ', fmt='%d')
    return s.getvalue().decode()


def _to_float_string(arr):
    s = io.BytesIO()
    np.savetxt(s, arr, delimiter=' ')
    return s.getvalue().decode()


@deprecated("skfem.visuals.matplotlib.plot (no replacement)")
def plot(basis, x, keys: Optional[str] = None):
    m = basis.mesh
    if isinstance(basis.elem, skfem.ElementVector):
        vdim = m.dim()
        x = x.reshape((-1, vdim)).flatten('F')
    else:
        vdim = 1
    bfacets = m.boundary_facets()
    nbfacets = len(bfacets)
    return glvis(template.format(
        m.dim(),
        m.nelements,
        _to_int_string(np.hstack((
            np.ones(m.nelements, dtype=np.int32)[:, None],
            (np.zeros(m.nelements, dtype=np.int32)[:, None]
             + MESH_TYPE_MAPPING[type(m)]),
            m.t.T,
        ))),
        nbfacets,
        _to_int_string(np.hstack((
            np.ones(nbfacets, dtype=np.int32)[:, None],
            (np.zeros(nbfacets, dtype=np.int32)[:, None]
             + BOUNDARY_TYPE_MAPPING[MESH_TYPE_MAPPING[type(m)]]),
            m.facets[:, bfacets].T,
        ))),
        m.nvertices,
        m.doflocs.shape[0],
        _to_float_string(m.doflocs.T),
        'Linear',
        vdim,
        _to_float_string(x[:, None]),
        "keys\n{}".format(keys) if keys is not None else "",
    ))


@deprecated("skfem.visuals.matplotlib.draw (no replacement)")
def draw(basis, keys: Optional[str] = None):
    return plot(basis, basis.zeros(), keys=keys)
