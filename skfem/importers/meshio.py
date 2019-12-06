"""Import any formats supported by meshio."""

from typing import Tuple
from collections import OrderedDict
import warnings

import numpy as np
import skfem

import meshio


MESH_TYPE_MAPPING = OrderedDict([
    ('tetra', skfem.MeshTet),
    ('hexahedron', skfem.MeshHex),
    ('triangle', skfem.MeshTri),
    ('quad', skfem.MeshQuad),
    ('line', skfem.MeshLine),
])


def from_meshio(m, force_mesh_type=None):
    """Convert meshio mesh into :class:`skfem.mesh.Mesh`.

    Parameters
    ----------
    m
        The mesh object from meshio.
    force_mesh_type
        An optional string forcing the mesh type if automatic detection
        fails. See skfem.importers.meshio.MESH_TYPE_MAPPING for possible values.

    """

    if force_mesh_type is None:
        for k, v in MESH_TYPE_MAPPING.items():
            # find first if match
            if k in m.cells:
                meshio_type, mesh_type = (k, MESH_TYPE_MAPPING[k])
                break
    else:
        meshio_type, mesh_type = (force_mesh_type,
                                  MESH_TYPE_MAPPING[force_mesh_type])

    def strip_extra_coordinates(p):
        if meshio_type == "line":
            return p[:, :1]
        if meshio_type in ("quad", "triangle"):
            return p[:, :2]
        return p

    # create p and t
    p = np.ascontiguousarray(strip_extra_coordinates(m.points).T)
    t = np.ascontiguousarray(m.cells[meshio_type].T)

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

        # find subdomains
        if meshio_type in m.cell_data and 'gmsh:physical' in m.cell_data[meshio_type]:
            elements_tag = m.cell_data[meshio_type]['gmsh:physical']

            subdomains = {}
            tags = np.unique(elements_tag)

            for tag in tags:
                t_set = np.nonzero(tag == elements_tag)[0]
                subdomains[find_tagname(tag)] = t_set

        # find tagged boundaries
        if bnd_type in m.cell_data and 'gmsh:physical' in m.cell_data[bnd_type]:
            facets = m.cells[bnd_type]
            facets_tag = m.cell_data[bnd_type]['gmsh:physical']
            bndfacets = mtmp.boundary_facets()

            # put meshio facets to dict
            dic = {tuple(np.sort(facets[i])): facets_tag[i]
                   for i in range(facets.shape[0])}

            # get index of corresponding Mesh.facets for each meshio
            # facet found in the dict
            index = np.array([[dic[tuple(np.sort(mtmp.facets[:, i]))], i]
                              for i in bndfacets
                              if tuple(np.sort(mtmp.facets[:, i])) in dic])

            # read meshio tag numbers and names
            tags = index[:, 0]
            boundaries = {}
            for tag in np.unique(tags):
                tagindex = np.nonzero(tags == tag)[0]
                boundaries[find_tagname(tag)] = index[tagindex, 1]

        mtmp.boundaries = boundaries
        mtmp.subdomains = subdomains

    except Exception as e:
        warnings.warn("Unable to load tagged boundaries/subdomains.")

    return mtmp


def from_file(filename):
    return from_meshio(meshio.read(filename))
