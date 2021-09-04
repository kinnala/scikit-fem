import numpy as np

from .mesh import Mesh
from .element import Element
from .assembly import CellBasis


def splitmesh(mesh, nsplits):

    from dataclasses import replace

    return [
        replace(
            mesh,
            doflocs=mesh.p,
            t=mesh.t[:, ix],
        ) for ix in np.array_split(np.arange(mesh.nelements, dtype=np.int64),
                                   nsplits)
    ]


def Basis(mesh,
          elem,
          mapping=None,
          intorder=None,
          elements=None,
          quadrature=None,
          use_dask=True):

    from itertools import cycle

    # TODO not supported yet
    assert mapping is None
    assert intorder is None
    assert elements is None
    assert quadrature is None

    if isinstance(mesh, Mesh):
        mesh = splitmesh(mesh, 4)

    if isinstance(elem, Element):
        elem = [elem]

    params = [arg for arg in zip(mesh, cycle(elem))]

    if use_dask:
        try:
            import dask
            dask.config.set(scheduler='threading')
            import dask.bag as db
            bag = db.from_sequence(params)
            return bag.map(lambda p: CellBasis(*p))
        except Exception as e:
            print(e)

    return [CellBasis(*p) for p in params]


def asm(form,
        bases,
        compute=True,
        use_dask=True):

    if use_dask:
        try:
            if isinstance(bases, list):
                import dask.bag as db
                bases = db.from_sequence(bases)

            c = bases.map(lambda basis: form.coo_data(basis)).sum()

            if not compute:
                return c
            out = c.compute(scheduler='threading')
            return out.tocsr()
        except Exception as e:
            print(e)

    return sum(map(lambda basis: form.coo_data(basis), bases)).tocsr()
