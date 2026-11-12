"""Distributed meshes for assembling and solving with petsc4py.

A serial script becomes distributed by passing the mesh through
:func:`distribute`::

    m = distribute(MeshTet().refined(6))
    basis = Basis(m, ElementTetP1())
    A = laplace.assemble(basis)  # petsc4py.PETSc.Mat
    b = unit_load.assemble(basis)  # petsc4py.PETSc.Vec
    x = solve(*enforce(A, b, D=basis.get_dofs()),
              solver=solver_petsc(ksp_type='cg', pc_type='gamg'))

Each MPI process gets a part of the mesh.  The bases, the forms and
:meth:`~skfem.mesh.Mesh.boundary_facets` work on the local part, while
the assembled matrices and vectors are distributed PETSc objects.  The
solution returned by :func:`~skfem.utils.solve` is again a local NumPy
array, so that the postprocessing works as in serial.  Functionals are
summed over all processes.  Use :func:`allreduce` for other reductions
of local arrays, e.g., ``allreduce(x.max(), op='max')``.

Not supported for distributed meshes are, e.g., high order (curved) and
periodic meshes, :class:`~skfem.assembly.InteriorFacetBasis` (DG jump
terms, facet-based error estimators), forms with two different bases
(rectangular matrices, :func:`~skfem.utils.bmat`), eigenvalue problems,
:func:`~skfem.utils.penalize`, :func:`~skfem.utils.mpc`, and refining
the mesh after distributing it.  Point evaluation, e.g.,
:meth:`~skfem.assembly.CellBasis.probes`, finds only the points inside
the local part of the mesh.

"""
from dataclasses import dataclass
from typing import Any, Callable, Optional, Union

import numpy as np
from numpy import ndarray

from skfem.mesh import Mesh


@dataclass
class DistributedMesh:
    """The relation of a local submesh to the global mesh.

    The arrays map local vertex, edge, facet and element indices to the
    corresponding indices in the global mesh.

    """

    nvertices: int
    nedges: int
    nfacets: int
    nelements: int
    vertices: ndarray
    edges: Optional[ndarray]
    facets: ndarray
    elements: ndarray
    boundary_facets: ndarray  #: local indices of the global boundary facets
    #: global index of the first element of each local facet, i.e.
    #: ``f2t[0]`` of the global mesh, for orienting the facets consistently
    facet_first: ndarray
    comm: Any = None

    def __getstate__(self):
        # communicators are not picklable
        return {**self.__dict__, 'comm': None}

    def check(self, mesh: Mesh):
        """Raise if the mesh topology has changed after distributing."""
        if (len(self.vertices) != mesh.nvertices
                or len(self.elements) != mesh.nelements):
            raise ValueError("The topology of a distributed mesh has been "
                             "modified after calling 'distribute'.  "
                             "Modify the mesh before distributing it.")


def _default_comm(comm=None):
    if comm is None:
        import petsc4py.PETSc as petsc
        comm = petsc.COMM_WORLD
    return comm


def _local_part(mesh: Mesh, mship: ndarray, rank: int) -> Mesh:
    """Restrict the mesh to a subdomain and record the global indices."""
    subix = np.nonzero(mship == rank)[0].astype(np.int32)
    sub, vertices = mesh.restrict(subix, return_mapping=True)

    facets = np.empty(sub.nfacets, dtype=np.int32)
    facets[sub.t2f] = mesh.t2f[:, subix]
    if not np.array_equal(np.sort(vertices[sub.facets], axis=0),
                          np.sort(mesh.facets[:, facets], axis=0)):
        raise NotImplementedError("Failed to map the local facets to "
                                  "the global facets.")

    edges = None
    if mesh.dim() == 3:
        edges = np.empty(sub.nedges, dtype=np.int32)
        edges[sub.t2e] = mesh.t2e[:, subix]

    is_boundary = np.zeros(mesh.nfacets, dtype=bool)
    is_boundary[mesh.boundary_facets()] = True

    sub._dist = DistributedMesh(
        nvertices=mesh.nvertices,
        nedges=mesh.nedges if mesh.dim() == 3 else 0,
        nfacets=mesh.nfacets,
        nelements=mesh.nelements,
        vertices=vertices.astype(np.int32),
        edges=edges,
        facets=facets,
        elements=subix,
        boundary_facets=np.nonzero(is_boundary[facets])[0].astype(np.int32),
        facet_first=mesh.f2t[0, facets].astype(np.int32),
    )
    return sub


def _partition(mesh: Mesh, nparts: int) -> ndarray:
    """Element-wise membership in subdomains using METIS."""
    if nparts == 1:
        return np.zeros(mesh.nelements, dtype=np.int32)
    from pymetis import part_mesh, GType
    _, mship, _ = part_mesh(
        nparts,
        mesh.t.T,
        gtype=GType.DUAL,
        ncommon=len(mesh.refdom.facets[0]),
    )
    return np.array(mship, dtype=np.int32)


def distribute(mesh: Union[Mesh, Callable[[], Mesh]], comm=None) -> Mesh:
    """Split the mesh between MPI processes and return the local part.

    The mesh is partitioned using METIS (``pymetis``) on the first
    process and each process receives its own part.  The resulting mesh
    can be used as usual, e.g., to create bases; then the forms assemble
    into distributed PETSc matrices and vectors.  Refining or otherwise
    modifying the topology of the result is not supported.

    Parameters
    ----------
    mesh
        The global mesh, or a function returning the global mesh.  A
        function is called only on the first process which saves memory
        and time for large meshes.
    comm
        The communicator, by default ``petsc4py.PETSc.COMM_WORLD``.

    """
    comm = _default_comm(comm)
    if getattr(mesh, '_dist', None) is not None:
        raise ValueError("The mesh is already distributed.")

    parts = None
    error = None
    if comm.rank == 0:
        if callable(mesh) and not isinstance(mesh, Mesh):
            mesh = mesh()
        if mesh.doflocs.shape[1] != mesh.nvertices:
            error = ("Distributing high order or periodic meshes is not "
                     "supported.")
    if comm.size > 1:
        error = comm.tompi4py().bcast(error, root=0)
    if error is not None:
        # raise on all processes to avoid a deadlock
        raise NotImplementedError(error)
    if comm.rank == 0:
        mship = _partition(mesh, comm.size)
        parts = [_local_part(mesh, mship, rank) for rank in range(comm.size)]

    if comm.size == 1:
        local = parts[0]
    else:
        local = comm.tompi4py().scatter(parts, root=0)
    local._dist.comm = comm
    return local


def allreduce(value: Any, op: str = 'sum', comm=None) -> Any:
    """Combine a local value over all processes.

    Useful for reductions of local arrays such as the solution returned
    by :func:`~skfem.utils.solve`.  Returns ``value`` as is if
    ``petsc4py`` is not installed or there is only one process.

    Parameters
    ----------
    value
        The local value, e.g., a scalar.
    op
        One of 'sum', 'max', 'min' or 'prod'.
    comm
        The communicator, by default ``petsc4py.PETSc.COMM_WORLD``.

    """
    try:
        comm = _default_comm(comm)
    except ImportError:
        return value
    if comm.size == 1:
        return value
    from mpi4py import MPI
    ops = {'sum': MPI.SUM, 'max': MPI.MAX, 'min': MPI.MIN, 'prod': MPI.PROD}
    return comm.tompi4py().allreduce(value, op=ops[op])


def is_petsc(obj: Any) -> bool:
    """Is the object from petsc4py; avoids importing petsc4py."""
    return type(obj).__module__.startswith('petsc4py')


def global_indices(A) -> ndarray:
    """The global indices of the local DOFs of an assembled matrix."""
    lgmap = A.getLGMap()[0]
    if lgmap.handle == 0:
        raise ValueError("The PETSc matrix has no local-to-global mapping: "
                         "use a matrix assembled with a distributed basis.")
    return lgmap.getIndices()


def to_petsc_vec(A, x: Optional[ndarray], ix: Optional[ndarray] = None):
    """Distributed vector, compatible with ``A``, from local values.

    Parameters
    ----------
    A
        Distributed PETSc matrix.
    x
        The local values, or zero if None.
    ix
        Optionally, set only these local DOFs; ``x`` has the same length.

    """
    import petsc4py.PETSc as petsc
    vec = A.createVecRight()
    vec.set(0.)
    if x is not None:
        glob = global_indices(A)
        vec.setValues(glob if ix is None else glob[ix], x,
                      addv=petsc.InsertMode.INSERT)
    vec.assemble()
    return vec


def to_local(A, vec) -> ndarray:
    """Local values of a distributed vector compatible with ``A``."""
    import petsc4py.PETSc as petsc
    ix = global_indices(A)
    iset = petsc.IS().createGeneral(ix, comm=petsc.COMM_SELF)
    out = petsc.Vec().createSeq(len(ix), comm=petsc.COMM_SELF)
    scatter = petsc.Scatter().create(vec, iset, out, None)
    scatter.scatter(vec, out, addv=petsc.InsertMode.INSERT)
    values = out.getArray().copy()
    for obj in (scatter, out, iset):
        obj.destroy()
    return values
