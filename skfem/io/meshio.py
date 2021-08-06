"""Import/export any formats supported by meshio."""

import meshio
import numpy as np

import skfem


MESH_TYPE_MAPPING = {
    'tetra': skfem.MeshTet,
    'hexahedron': skfem.MeshHex,
    'triangle': skfem.MeshTri,
    'quad': skfem.MeshQuad,
    'line': skfem.MeshLine,
    'tetra10': skfem.MeshTet2,
    'triangle6': skfem.MeshTri2,
    'quad9': skfem.MeshQuad2,
}

BOUNDARY_TYPE_MAPPING = {
    'line': 'vertex',
    'triangle': 'line',
    'quad': 'line',
    'tetra': 'triangle',
    'hexahedron': 'quad',
    'tetra10': 'triangle',  # TODO support quadratic facets
    'triangle6': 'line',  # TODO
    'quad9': 'line',  # TODO
}

TYPE_MESH_MAPPING = {MESH_TYPE_MAPPING[k]: k
                     for k in dict(reversed(list(MESH_TYPE_MAPPING.items())))}


def from_meshio(m,
                int_data_to_sets=False):
    """Convert meshio mesh into :class:`skfem.mesh.Mesh`.

    Parameters
    ----------
    m
        The mesh from meshio.
    int_data_to_sets
        We correctly read the so-called "cell sets" from ``meshio``.  However,
        many mesh formats do not support "cell sets" natively and, instead, use
        cellwise integer data to distinguish between different subdomains and
        boundaries.  If ``True``, call ``meshio.Mesh.int_data_to_sets`` to
        convert between the representations before attempting to read tags from
        ``meshio``.

    Returns
    -------
    A :class:`~skfem.mesh.Mesh` object.

    """
    cells = m.cells_dict
    meshio_type = None

    # detect 3D
    for k in cells:
        if k in {'tetra', 'hexahedron', 'tetra10'}:
            meshio_type = k
            break

    if meshio_type is None:
        # detect 2D
        for k in cells:
            if k in {'triangle', 'quad', 'triangle6', 'quad9'}:
                meshio_type = k
                break

    if meshio_type is None:
        # detect 1D
        for k in cells:
            if k == 'line':
                meshio_type = k
                break

    if meshio_type is None:
        raise NotImplementedError("Mesh type(s) not supported "
                                  "in import: {}.".format(cells.keys()))

    mesh_type = MESH_TYPE_MAPPING[meshio_type]

    # create p and t
    p = np.ascontiguousarray(mesh_type.strip_extra_coordinates(m.points).T)
    t = np.ascontiguousarray(cells[meshio_type].T)

    # reorder t if needed
    if meshio_type == 'hexahedron':
        t = t[[0, 4, 3, 1, 7, 5, 2, 6]]

    if int_data_to_sets:
        m.int_data_to_sets()

    subdomains = None
    boundaries = None

    # parse any subdomains from cell_sets
    if m.cell_sets:
        subdomains = {k: v[meshio_type]
                      for k, v in m.cell_sets_dict.items()
                      if meshio_type in v}

    # create temporary mesh for matching boundary elements
    mtmp = mesh_type(p, t)
    bnd_type = BOUNDARY_TYPE_MAPPING[meshio_type]

    # parse boundaries from cell_sets
    if m.cell_sets and bnd_type in m.cells_dict:
        facets = {k: [tuple(f) for f in
                      np.sort(m.cells_dict[bnd_type][v[bnd_type]])]
                  for k, v in m.cell_sets_dict.items()
                  if bnd_type in v}
        boundaries = {k: np.array([i for i, f in
                                   enumerate(map(tuple, mtmp.facets.T))
                                   if f in v])
                      for k, v in facets.items()}

    # MSH 2.2 tag parsing
    if m.cell_data and m.field_data:

        try:

            def find_tagname(tag):
                for key in m.field_data:
                    if m.field_data[key][0] == tag:
                        return key
                return None

            elements_tag = m.cell_data_dict['gmsh:physical'][meshio_type]
            subdomains = {}
            tags = np.unique(elements_tag)

            for tag in tags:
                t_set = np.nonzero(tag == elements_tag)[0]
                subdomains[find_tagname(tag)] = t_set

            # find tagged boundaries
            if bnd_type in m.cell_data_dict['gmsh:physical']:
                facets = m.cells_dict[bnd_type]
                facets_tag = m.cell_data_dict['gmsh:physical'][bnd_type]

            # put meshio facets to dict
            dic = {tuple(np.sort(facets[i])): facets_tag[i]
                   for i in range(facets.shape[0])}

            # get index of corresponding Mesh.facets for each meshio
            # facet found in the dict
            index = np.array([[dic[tuple(np.sort(mtmp.facets[:, i]))], i]
                              for i in mtmp.boundary_facets()
                              if tuple(np.sort(mtmp.facets[:, i])) in dic])

            # read meshio tag numbers and names
            tags = index[:, 0]
            boundaries = {}
            for tag in np.unique(tags):
                tagindex = np.nonzero(tags == tag)[0]
                boundaries[find_tagname(tag)] = index[tagindex, 1]

        except Exception:
            pass

    # attempt parsing tags from cell_data
    if m.cell_data:

        try:
            for k in m.cell_data:
                if k.startswith('skfem:'):
                    boundaries, subdomains = mtmp.decode_cell_data(m.cell_data)
                else:
                    # some mesh formats cannot preserve tag names
                    # => invent names for the tags
                    tags = 'skfem:' + ["set{}".format(i) for i in range(20)]
                    boundaries, subdomains = mtmp.decode_cell_data({
                        tags: m.cell_data[k]
                    })
                break

        except Exception:
            pass

    mtmp = mesh_type(p, t, boundaries, subdomains)

    return mtmp


def from_file(filename, **kwargs):
    return from_meshio(meshio.read(filename), **kwargs)


def to_meshio(mesh,
              point_data=None,
              cell_data=None):

    t = mesh.dofs.element_dofs.copy()
    if isinstance(mesh, skfem.MeshHex):
        t = t[[0, 3, 6, 2, 1, 5, 7, 4]]

    mtype = TYPE_MESH_MAPPING[type(mesh)]
    bmtype = BOUNDARY_TYPE_MAPPING[mtype]
    cells = {
        mtype: t.T,
        bmtype: mesh.facets[:, mesh.boundary_facets()].T
    }

    if cell_data is None:
        cell_data = {}

    cell_data.update(mesh.encode_cell_data())

    if point_data is None:
        point_data = {}

    point_data.update(mesh.encode_point_data())

    mio = meshio.Mesh(
        mesh.p.T,
        cells,
        point_data=point_data,
        cell_data=cell_data,
    )

    return mio


def to_file(mesh,
            filename,
            point_data=None,
            cell_data=None,
            **kwargs):

    meshio.write(filename,
                 to_meshio(mesh,
                           point_data,
                           cell_data),
                 **kwargs)
