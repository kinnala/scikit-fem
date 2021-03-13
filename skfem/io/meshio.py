"""Import any formats supported by meshio."""

import warnings

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

TYPE_MESH_MAPPING = {v: k for k, v in MESH_TYPE_MAPPING.items()}


def from_meshio(m, force_mesh_type=None):
    """Convert meshio mesh into :class:`skfem.mesh.Mesh`.

    Parameters
    ----------
    m
        The mesh from meshio.
    force_mesh_type
        An optional string forcing the mesh type if automatic detection
        fails. See :data:`skfem.io.meshio.MESH_TYPE_MAPPING` for possible
        values.

    Returns
    -------
    A :class:`~skfem.mesh.Mesh` object.

    """

    cells = m.cells_dict

    if force_mesh_type is None:
        meshio_type = None

        for k, v in MESH_TYPE_MAPPING.items():
            # find first if match
            if k in cells:
                meshio_type, mesh_type = k, v
                break

        if meshio_type is None:
            raise NotImplementedError("Mesh type(s) not supported "
                                      "in import: {}.".format(cells.keys()))
    else:
        meshio_type, mesh_type = (force_mesh_type,
                                  MESH_TYPE_MAPPING[force_mesh_type])

    # create p and t
    p = np.ascontiguousarray(mesh_type.strip_extra_coordinates(m.points).T)
    t = np.ascontiguousarray(cells[meshio_type].T)

    # reorder t if needed
    if meshio_type == 'hexahedron':
        t = t[[0, 4, 3, 1, 7, 5, 2, 6]]

    mtmp = mesh_type(p, t)

    try:
        # element to boundary element type mapping
        bnd_type = {
            'line': 'vertex',
            'triangle': 'line',
            'quad': 'line',
            'tetra': 'triangle',
            'hexahedron': 'quad',
        }[meshio_type]

        def find_tagname(tag):
            for key in m.field_data:
                if m.field_data[key][0] == tag:
                    return key
            return None

        if m.cell_sets:  # MSH 4.1
            subdomains = {k: v[meshio_type]
                          for k, v in m.cell_sets_dict.items()
                          if meshio_type in v}
            facets = {k: [tuple(f) for f in
                          np.sort(m.cells_dict[bnd_type][v[bnd_type]])]
                      for k, v in m.cell_sets_dict.items()
                      if bnd_type in v}
            boundaries = {k: np.array([i for i, f in
                                       enumerate(map(tuple, mtmp.facets.T))
                                       if f in v])
                          for k, v in facets.items()}
        else:  # MSH 2.2?
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

        mtmp = mesh_type(p, t, boundaries, subdomains)

    except Exception as e:
        warnings.warn("Unable to load tagged boundaries/subdomains.")
        print(e)

    return mtmp


def from_file(filename):
    return from_meshio(meshio.read(filename))


def to_meshio(mesh, point_data=None):

    t = mesh.t.copy()
    if isinstance(mesh, skfem.MeshHex):
        t = t[[0, 3, 6, 2, 1, 5, 7, 4]]

    cells = {TYPE_MESH_MAPPING[type(mesh)]: t.T}
    return meshio.Mesh(mesh.p.T, cells, point_data)


def to_file(mesh, filename, point_data=None, **kwargs):
    meshio.write(filename, to_meshio(mesh, point_data), **kwargs)
