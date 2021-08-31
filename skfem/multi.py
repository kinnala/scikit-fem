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


def MultiBasis(mesh,
               elem,
               mapping=None,
               intorder=None,
               elements=None,
               quadrature=None,
               use_dask=False):

    from itertools import cycle

    # TODO not supported yet
    assert mapping is None
    assert intorder is None
    assert elements is None
    assert quadrature is None

    if isinstance(mesh, Mesh):
        mesh = [mesh]

    if isinstance(elem, Element):
        elem = [elem]

    params = [arg for arg in zip(mesh, cycle(elem))]

    if use_dask:
        import dask.bag as db
        bag = db.from_sequence(params)
        return bag.map(lambda p: Basis(*p))

    return [Basis(*p) for p in params]


def multiasm(form, bases):

    if not isinstance(bases, list):
        c = bases.map(lambda basis: form.coo_data(basis)).sum()
        out = c.compute(scheduler='threading')
        return out.tocsr()

    return sum(map(lambda basis: form.coo_data(basis), bases)).tocsr()
